"""Live adapters: existing vision UDP/HTTP and robot BLE behind domain APIs."""
import asyncio
from dataclasses import replace
import json
import logging
import re
import socket
import struct
from threading import Event, Thread
import time
from urllib.request import build_opener, ProxyHandler

from .components import CarCmd, GPSState, rpc_method
from .events import EventHub
from .models import BallState, CarState, Field, Pose, WorldState
from .provider import VSSSProvider

log = logging.getLogger(__name__)


class GPSLive(GPSState):
    def __init__(self, events, *, server="192.168.12.1", port=5000,
                 listen_port=0, bind="0.0.0.0", metadata_url=None):
        super().__init__(events)
        self.server, self.port = server, port
        self.metadata_url = metadata_url or (f"http://{server}:8080/api/v1/status" if server else None)
        self._stop = Event()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((bind, listen_port))
        self._socket.settimeout(.2)
        self.listen_port = self._socket.getsockname()[1]
        self._thread = Thread(target=self._receive, daemon=True)
        self._metadata_thread = Thread(target=self._initial_field, daemon=True)
        self._thread.start()
        self._metadata_thread.start()

    def _initial_field(self):
        try:
            self.refresh_field()
        except Exception as error:
            self.events.publish("lifecycle", {"kind": "field_config_error", "error": str(error)})

    @rpc_method
    def refresh_field(self):
        if not self.metadata_url:
            return self.state().field
        opener = build_opener(ProxyHandler({}))
        with opener.open(self.metadata_url, timeout=2) as response:
            data = json.loads(response.read(1024 * 1024))
        width, height = data["setup"]["board"]["dimensions_m"]
        field = Field(width, height)
        with self._lock:
            self._state = replace(self._state, field=field)
            state = self.state()
        self.events.publish("gps", state)
        return field

    def _decode(self, payload):
        packet = json.loads(payload)
        if type(packet.get("board_detected")) is not bool:
            raise ValueError("invalid board detection flag")
        return WorldState(field=self.state().field,
            cars=[CarState(int(c["id"]), Pose(c["x"], c["y"], c["angle"])) for c in packet["markers"]],
            balls=[BallState(i, Pose(b["x"], b["y"])) for i, b in enumerate(packet.get("balls", []))],
            captured_at=packet.get("timestamp", time.time()), sequence=packet.get("frame_sequence", 0),
            valid=packet["board_detected"])

    def _receive(self):
        renewal = 0.
        while not self._stop.is_set():
            try:
                if self.server and time.monotonic() >= renewal:
                    self._socket.sendto(b"POSE_SUBSCRIBE", (self.server, self.port + 1))
                    renewal = time.monotonic() + 1
                payload, _ = self._socket.recvfrom(65535)
                self._update(self._decode(payload))
            except socket.timeout:
                continue
            except (ValueError, KeyError, TypeError, OverflowError) as error:
                self.events.publish("lifecycle", {"kind": "telemetry_error", "error": str(error)})
            except OSError:
                if not self._stop.is_set():
                    log.exception("vision reception stopped")
                return

    def close(self):
        self._stop.set()
        if self.server:
            try:
                self._socket.sendto(b"POSE_UNSUBSCRIBE", (self.server, self.port + 1))
            except OSError:
                pass
        self._socket.close()
        self._thread.join(timeout=3)
        self._metadata_thread.join(timeout=3)


# Firmware details are deliberately confined to this adapter.
VELOCITY_UUID = "12345678-1234-5678-1234-56789abcdef1"
STATUS_UUID = "12345678-1234-5678-1234-56789abcdef9"


class CarCmdLive(CarCmd):
    """One asyncio/BLE owner thread; lazy connections, no subprocess wire protocol."""
    def __init__(self, events):
        super().__init__(events)
        self._loop = asyncio.new_event_loop()
        self._clients = {}
        self._active = {}
        self._ready = Event()
        self._closing = False

        def run():
            asyncio.set_event_loop(self._loop)
            self._io_lock = asyncio.Lock()
            self._heartbeat = self._loop.create_task(self._keepalive())
            self._ready.set()
            self._loop.run_forever()
            self._loop.close()
        self._thread = Thread(target=run, daemon=True)
        self._thread.start()
        self._ready.wait()

    def _call(self, coroutine):
        if self._closing:
            coroutine.close()
            raise RuntimeError("BLE adapter is closed")
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result(timeout=20)
        except TimeoutError:
            future.cancel()
            raise TimeoutError("BLE operation timed out; result may be uncertain") from None

    async def _scan(self):
        from bleak import BleakScanner
        devices = await BleakScanner.discover(timeout=4, return_adv=True)
        return sorted({int(match.group(1)) for device, advertisement in devices.values()
                       if (match := re.fullmatch(r"car(\d+)", advertisement.local_name or device.name or ""))})

    @rpc_method
    def available(self):
        return self._call(self._scan())

    async def _connect(self, car_id):
        from bleak import BleakClient, BleakScanner
        client = self._clients.get(car_id)
        if client and client.is_connected:
            return client
        self._active.pop(car_id, None)  # Never restore motion after reconnect.
        device = await BleakScanner.find_device_by_filter(
            lambda device, adv: (adv.local_name or device.name) == f"car{car_id}", timeout=5)
        if device is None:
            raise ConnectionError(f"car{car_id} not found")
        client = BleakClient(device, timeout=8)
        try:
            await client.connect()
            await asyncio.wait_for(client.write_gatt_char(VELOCITY_UUID, bytes(3), response=True), 2)
        except Exception:
            if client.is_connected:
                await client.disconnect()
            raise
        self._clients[car_id] = client
        self._active[car_id] = bytes(3)
        return client

    @staticmethod
    def _payload(velocity):
        return struct.pack("<bbb", *(max(-128, min(127, round(v * 128))) for v in (velocity.x, velocity.y, velocity.w)))

    async def _write(self, car_id, velocity):
        async with self._io_lock:
            client = await self._connect(car_id)
            payload = self._payload(velocity)
            try:
                await asyncio.wait_for(client.write_gatt_char(VELOCITY_UUID, payload, response=True), 2)
            except Exception:
                self._active.pop(car_id, None)
                await client.disconnect()
                raise
            self._active[car_id] = payload

    def _apply(self, car_id, velocity):
        self._call(self._write(car_id, velocity))

    async def _status(self, car_id):
        async with self._io_lock:
            client = await self._connect(car_id)
            data = await asyncio.wait_for(client.read_gatt_char(STATUS_UUID), 2)
            mode, flags, fault, _, x, y, angle, count = struct.unpack("<BBBBfffI", data)
            return {"mode": mode, "wifi_connected": bool(flags & 1), "pose_fresh": bool(flags & 2),
                    "fault": fault, "pose": Pose(x, y, angle), "packet_count": count}

    @rpc_method
    def status(self, car_id):
        if type(car_id) is not int or car_id < 0:
            raise ValueError("invalid car ID")
        return self._call(self._status(car_id))

    async def _keepalive(self):
        try:
            while True:
                await asyncio.sleep(.2)
                async with self._io_lock:
                    for car_id, payload in tuple(self._active.items()):
                        try:
                            client = self._clients[car_id]
                            await asyncio.wait_for(client.write_gatt_char(VELOCITY_UUID, payload, response=True), 1)
                        except Exception as error:
                            self._active.pop(car_id, None)
                            self.events.publish("lifecycle", {"kind": "car_disconnected", "car_id": car_id, "error": str(error)})
        except asyncio.CancelledError:
            return

    async def _shutdown(self):
        self._heartbeat.cancel()
        await self._heartbeat
        async with self._io_lock:
            for client in self._clients.values():
                try:
                    if client.is_connected:
                        await asyncio.wait_for(client.write_gatt_char(VELOCITY_UUID, bytes(3), response=True), 1)
                        await client.disconnect()
                except Exception:
                    log.exception("BLE shutdown failed; firmware lease expiry remains active")

    def close(self):
        if self._closing:
            return
        try:
            self._call(self._shutdown())
        finally:
            self._closing = True
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=3)
            self._owners.clear()


def main(argv=None):
    from .cli import provider_parser, run_provider
    parser = provider_parser(__doc__)
    parser.add_argument("--vision-server", default="192.168.12.1")
    parser.add_argument("--listen-port", type=int, default=0)
    args = parser.parse_args(argv)
    events = EventHub()
    gps = GPSLive(events, server=args.vision_server or None, listen_port=args.listen_port)
    run_provider(VSSSProvider(gps, CarCmdLive(events)), args)


if __name__ == "__main__":
    main()
