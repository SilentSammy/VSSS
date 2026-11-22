"""Live adapters: existing vision UDP/HTTP and robot BLE behind domain APIs."""
import asyncio
from dataclasses import replace
import json
import logging
import math
import re
import socket
import struct
from threading import Event, Lock, Thread
import time
from urllib.request import build_opener, ProxyHandler

from .components import CarCmd, GPSState, rpc_method
from .events import EventHub
from .models import BallState, CarState, Field, Pose, WorldState
from .provider import VSSSProvider
from . import _ble_protocol as protocol

log = logging.getLogger(__name__)


class GPSLive(GPSState):
    def __init__(self, events, *, server="192.168.12.1", port=5000,
                 listen_port=0, bind="0.0.0.0", metadata_url=None, stale_after=.5):
        super().__init__(events, stale_after=stale_after)
        self._last_stamp = self._best_offset = self._stream = None
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
        board = data["setup"]["board"]
        # New vision metadata distinguishes the printable field from the ArUco
        # grid. Older servers expose only the latter.
        width, height = board.get("print_dimensions_m", board["dimensions_m"])
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
        age = packet.get('frame_age_ms', 0.)
        return WorldState(field=self.snapshot().field,
            cars=[CarState(c["id"], Pose(c["x"], c["y"], c["angle"])) for c in packet["markers"]],
            balls=[BallState(i, Pose(b["x"], b["y"])) for i, b in enumerate(packet.get("balls", []))],
            captured_at=packet.get("timestamp", time.time()), sequence=packet.get("frame_sequence", 0),
            valid=packet["board_detected"], frame_age_ms=age,
            received_at=time.time())

    def ingest(self, payload):
        """Validate source timing before presenting a packet as current."""
        packet = json.loads(payload)
        state = self._decode(payload)
        stream = packet.get('stream_id')
        if stream != self._stream:
            self._stream, self._last_stamp, self._best_offset = stream, None, None
        stamp = packet.get('timestamp')
        excess = 0.
        if stamp is not None:
            if self._last_stamp is not None and stamp < self._last_stamp:
                with self._lock: self._problem = 'Camera packets are out of order'
                return False
            offset = time.monotonic() - stamp
            self._best_offset = offset if self._best_offset is None else min(self._best_offset, offset)
            excess = max(0., offset - self._best_offset)
        age = state.frame_age_ms / 1000 + excess
        if age > self.stale_after:
            with self._lock: self._problem = 'Camera packets are delayed'
            return False
        if stamp is not None: self._last_stamp = stamp
        self._update(replace(state, source_age=age))
        return True

    def _receive(self):
        renewal = 0.
        while not self._stop.is_set():
            try:
                if self.server and time.monotonic() >= renewal:
                    self._socket.sendto(b"POSE_SUBSCRIBE", (self.server, self.port + 1))
                    renewal = time.monotonic() + 1
                payload, _ = self._socket.recvfrom(65535)
                self.ingest(payload)
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
        self._last_write = {}
        self._position_mode = set()
        self._generations = {}
        self._ready = Event()
        self._closing = False
        self._discovered = None
        self._discovered_at = 0.
        self._discovery_lock = Lock()

        def run():
            asyncio.set_event_loop(self._loop)
            self._io_locks = {}
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
    def available(self, refresh=False):
        # Scanning must not hold the command/ownership lock for four seconds.
        with self._discovery_lock:
            if refresh or self._discovered is None or time.monotonic() - self._discovered_at >= 30:
                self._discovered = self._call(self._scan())
                self._discovered_at = time.monotonic()
            return sorted(set(self._discovered) | {car_id for car_id, client in tuple(self._clients.items())
                                                       if client.is_connected})

    def _io_lock(self, car_id):
        # Called only on the asyncio thread. GATT operations serialize per car.
        return self._io_locks.setdefault(car_id, asyncio.Lock())

    async def _connect(self, car_id):
        from bleak import BleakClient, BleakScanner
        client = self._clients.get(car_id)
        if client and client.is_connected and car_id in self._active:
            return client
        if client and client.is_connected:
            try: await asyncio.wait_for(client.disconnect(), 2)
            except Exception: pass
        self._clients.pop(car_id, None)
        self._active.pop(car_id, None)  # Never restore motion after reconnect.
        self._position_mode.discard(car_id)
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
        self._generations[car_id] = self._generations.get(car_id, 0) + 1
        self._active[car_id] = bytes(3)
        self._last_write[car_id] = time.monotonic()
        return client

    @staticmethod
    def _payload(velocity):
        return struct.pack("<bbb", *(max(-128, min(127, round(v * 128))) for v in (velocity.x, velocity.y, velocity.w)))

    async def _write(self, car_id, velocity, generation=None):
        async with self._io_lock(car_id):
            if generation is None:
                client = await self._connect(car_id)
            else:
                client = self._clients.get(car_id)
                if (not client or not client.is_connected or car_id not in self._active or
                        self._generations.get(car_id) != generation):
                    raise ConnectionError(f'car{car_id} connection changed')
            payload = self._payload(velocity)
            try:
                await asyncio.wait_for(client.write_gatt_char(VELOCITY_UUID, payload, response=True), 2)
            except Exception:
                self._active.pop(car_id, None)
                await client.disconnect()
                raise
            self._active[car_id] = payload
            self._position_mode.discard(car_id)
            self._last_write[car_id] = time.monotonic()

    def _apply(self, car_id, velocity):
        self._call(self._write(car_id, velocity))

    def _apply_guarded(self, car_id, velocity, generation):
        self._call(self._write(car_id, velocity, generation))

    async def _send_waypoint(self, car_id, target):
        async with self._io_lock(car_id):
            client = await self._connect(car_id)
            payload = protocol.WAYPOINT_STRUCT.pack(target.x, target.y, math.radians(target.angle))
            try:
                await asyncio.wait_for(client.write_gatt_char(protocol.WAYPOINT_UUID, payload, response=True), 2)
            except Exception:
                self._active.pop(car_id, None)
                self._position_mode.discard(car_id)
                await client.disconnect()
                raise
            self._position_mode.add(car_id)
            self._last_write[car_id] = time.monotonic()

    def _waypoint(self, car_id, target):
        self._call(self._send_waypoint(car_id, target))

    async def _status(self, car_id):
        async with self._io_lock(car_id):
            client = await self._connect(car_id)
            data = await asyncio.wait_for(client.read_gatt_char(STATUS_UUID), 2)
            return protocol.RobotStatus(*protocol.STATUS_STRUCT.unpack(data))

    @rpc_method
    def status(self, car_id):
        if type(car_id) is not int or car_id < 0:
            raise ValueError("invalid car ID")
        return self._settings(car_id, 'status')

    def _operation(self, car_id, operation, value):
        if operation == 'connect': self._call(self._ensure_connected(car_id))
        if operation in ('connection', 'connect'):
            client = self._clients.get(car_id)
            return dict(connected=bool(client and client.is_connected and car_id in self._active),
                        generation=self._generations.get(car_id, 0))
        if operation == 'status': return self._call(self._status(car_id))
        return self._call(self._configure(car_id, operation, value))

    async def _ensure_connected(self, car_id):
        async with self._io_lock(car_id): await self._connect(car_id)

    async def _configure(self, car_id, operation, value):
        async with self._io_lock(car_id):
            client = await self._connect(car_id)
            async def read(uuid):
                return bytes(await asyncio.wait_for(client.read_gatt_char(uuid), 2))
            async def write(uuid, payload):
                await asyncio.wait_for(client.write_gatt_char(uuid, payload, response=True), 2)
            if operation in ('read_trim', 'set_trim'):
                if operation == 'set_trim':
                    payload = protocol.TRIM_STRUCT.pack(*(v for row in value for v in row))
                    await write(protocol.TRIM_MATRIX_UUID, payload)
                actual = await read(protocol.TRIM_MATRIX_UUID)
                if operation == 'set_trim' and actual != payload: raise ValueError('Firmware rejected trim')
                values = protocol.TRIM_STRUCT.unpack(actual)
                return [list(values[i:i+3]) for i in (0, 3, 6)]
            if operation in ('read_controller', 'set_controller'):
                actual = await read(protocol.CONTROLLER_UUID)
                values = dict(zip(protocol.CONTROLLER_KEYS, protocol.CONTROLLER_STRUCT.unpack(actual)))
                if operation == 'set_controller':
                    values = protocol.normalize_controller(value, values)
                    payload = protocol.CONTROLLER_STRUCT.pack(*(values[k] for k in protocol.CONTROLLER_KEYS))
                    await write(protocol.CONTROLLER_UUID, payload)
                    actual = await read(protocol.CONTROLLER_UUID)
                    if actual != payload: raise ValueError('Firmware rejected controller')
                return dict(zip(protocol.CONTROLLER_KEYS, protocol.CONTROLLER_STRUCT.unpack(actual)))
            if operation in ('read_robot_config', 'set_robot_config'):
                if operation == 'set_robot_config':
                    payload = json.dumps(value, separators=(',', ':'), allow_nan=False).encode()
                    for offset in range(0, len(payload), protocol.CONFIG_CHUNK_BYTES):
                        await write(protocol.CONFIG_UUID, b'\x01' + struct.pack('<HH', offset, len(payload)) +
                                    payload[offset:offset+protocol.CONFIG_CHUNK_BYTES])
                actual = protocol.normalize_robot_config(json.loads((await read(protocol.CONFIG_UUID)).decode()))
                if operation == 'set_robot_config':
                    if actual != value: raise ValueError('Firmware rejected robot configuration')
                    await self._reset_link(car_id, client, write, protocol.CONTROL_RESTART)
                return actual
            if operation in ('restart', 'reconnect_telemetry'):
                code = protocol.CONTROL_RESTART if operation == 'restart' else protocol.CONTROL_RECONNECT_TELEMETRY
                await write(protocol.CONTROL_UUID, bytes([code]))
                if await read(protocol.CONTROL_UUID) != bytes([0xa5, code]):
                    raise RuntimeError('Firmware rejected recovery command')
                self._active.pop(car_id, None)
                self._position_mode.discard(car_id)
                if operation == 'restart':
                    await asyncio.wait_for(client.disconnect(), 2)
                    self._clients.pop(car_id, None)
                return True
            raise ValueError('Unknown robot operation')

    async def _reset_link(self, car_id, client, write, code):
        await write(protocol.CONTROL_UUID, bytes([code]))
        self._active.pop(car_id, None)
        self._position_mode.discard(car_id)
        await asyncio.wait_for(client.disconnect(), 2)
        self._clients.pop(car_id, None)

    async def _keepalive(self):
        try:
            while True:
                await asyncio.sleep(.02)
                due = [car_id for car_id in self._active
                       if time.monotonic() - self._last_write.get(car_id, 0) >= .2]
                await asyncio.gather(*(self._heartbeat_car(car_id) for car_id in due))
        except asyncio.CancelledError:
            return

    async def _heartbeat_car(self, car_id):
        async with self._io_lock(car_id):
            # A command may have completed while we waited for this lock.
            # Fresh writes renew the lease themselves; do not repeat them.
            if car_id not in self._active or time.monotonic() - self._last_write.get(car_id, 0) < .2:
                return
            try:
                client = self._clients[car_id]
                position = car_id in self._position_mode
                payload = b'\x01' if position else self._active[car_id]
                uuid = protocol.LEASE_UUID if position else VELOCITY_UUID
                await asyncio.wait_for(client.write_gatt_char(uuid, payload, response=True), 1)
                self._last_write[car_id] = time.monotonic()
            except Exception as error:
                self._active.pop(car_id, None)
                self.events.publish("lifecycle", {"kind": "car_disconnected", "car_id": car_id, "error": str(error)})

    async def _shutdown(self):
        self._heartbeat.cancel()
        await self._heartbeat
        for car_id, client in tuple(self._clients.items()):
            async with self._io_lock(car_id):
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
