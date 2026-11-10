"""BLE owner process. stdin carries commands; stdout carries status messages.

The student script owns stdin's other end. EOF means that script has exited,
including os._exit() or an IDE stop, so we stop and disconnect immediately.
"""
import asyncio
import json
import struct
import sys
from threading import Thread
from bleak import BleakClient, BleakScanner

VELOCITY_UUID = '12345678-1234-5678-1234-56789abcdef1'
LEASE_UUID = '12345678-1234-5678-1234-56789abcdef2'
TRIM_MATRIX_UUID = '12345678-1234-5678-1234-56789abcdef3'
WAYPOINT_UUID = '12345678-1234-5678-1234-56789abcdef6'
HEARTBEAT_SECONDS = 0.2
WRITE_TIMEOUT = 1.0


class Commands:
    """Only keep the newest velocity, rather than queueing old movements."""
    def __init__(self):
        self.closed = asyncio.Event()
        self.changed = asyncio.Event()
        self.ready = False
        self.velocity = bytes(3)
        self.sequence = 0
        self.force = False
        self.trims = asyncio.Queue()
        self.waypoints = asyncio.Queue()
        self.position_mode = False

    def update(self, message):
        if self.ready:
            if 'trim_matrix' in message:
                self.trims.put_nowait(message)
                self.changed.set()
                return
            if 'waypoint' in message:
                self.waypoints.put_nowait(message)
                self.position_mode = True
                self.changed.set()
                return
            self.velocity = bytes(message['velocity'])
            self.sequence = message['sequence']
            self.force = message['force']
            self.position_mode = False
            self.changed.set()

    def report(self, **message):
        try:
            print(json.dumps(message), flush=True)
        except (BrokenPipeError, OSError):
            self.closed.set()


def read_commands(loop, commands):
    try:
        for line in sys.stdin:
            loop.call_soon_threadsafe(commands.update, json.loads(line))
    finally:
        # This thread is only a pipe reader; all BLE work stays on asyncio.
        try:
            loop.call_soon_threadsafe(commands.closed.set)
        except RuntimeError:
            pass  # The event loop has already finished shutting down.


async def write(client, velocity):
    await asyncio.wait_for(
        client.write_gatt_char(VELOCITY_UUID, velocity, response=True), WRITE_TIMEOUT)


async def control(client, commands):
    # Never resume an old movement after reconnecting.
    commands.velocity = bytes(3)
    commands.sequence = 0
    commands.force = False
    commands.position_mode = False
    await write(client, bytes(3))
    while not commands.trims.empty():
        commands.report(trim_done=commands.trims.get_nowait()['request'],
                        trim_error='Connection lost before applying trim')
    while not commands.waypoints.empty():
        commands.report(waypoint_done=commands.waypoints.get_nowait()['request'],
                        waypoint_error='Connection lost before applying waypoint')
    try:
        matrix = struct.unpack('<9f', await asyncio.wait_for(client.read_gatt_char(TRIM_MATRIX_UUID), WRITE_TIMEOUT))
    except Exception:
        matrix = None  # Driving is still possible; configuring trim requires matrix firmware.
    matrix_supported = matrix is not None
    commands.report(trim_matrix=matrix)
    commands.ready = True
    commands.report(ready=True, message='Connected')
    last_velocity = bytes(3)
    last_sequence = 0
    last_write = asyncio.get_running_loop().time()
    while client.is_connected:
        commands.changed.clear()
        velocity, sequence = commands.velocity, commands.sequence
        elapsed = asyncio.get_running_loop().time() - last_write
        if commands.position_mode:
            if elapsed >= HEARTBEAT_SECONDS:
                await asyncio.wait_for(
                    client.write_gatt_char(LEASE_UUID, bytes(1), response=True), WRITE_TIMEOUT)
                last_write = asyncio.get_running_loop().time()
        elif (velocity != last_velocity or elapsed >= HEARTBEAT_SECONDS
              or (commands.force and sequence != last_sequence)):
            await write(client, velocity)
            last_velocity, last_sequence = velocity, sequence
            last_write = asyncio.get_running_loop().time()
            commands.report(written=sequence)
        if not commands.trims.empty():
            message = commands.trims.get_nowait()
            try:
                if not matrix_supported:
                    raise RuntimeError('Firmware does not support matrix trim; update the ESP32 firmware')
                payload = struct.pack('<9f', *message['trim_matrix'])
                await asyncio.wait_for(client.write_gatt_char(TRIM_MATRIX_UUID, payload, response=True), WRITE_TIMEOUT)
                actual = await asyncio.wait_for(client.read_gatt_char(TRIM_MATRIX_UUID), WRITE_TIMEOUT)
                if bytes(actual) != payload:
                    raise ValueError('Firmware rejected trim')
                commands.report(trim_matrix=struct.unpack('<9f', actual), trim_done=message['request'])
            except Exception as error:
                commands.report(trim_done=message['request'], trim_error=str(error))
                raise  # Reconnect stopped if a configuration operation stalled.
        if not commands.waypoints.empty():
            message = commands.waypoints.get_nowait()
            try:
                payload = struct.pack('<fff', *message['waypoint'])
                await asyncio.wait_for(
                    client.write_gatt_char(WAYPOINT_UUID, payload, response=True), WRITE_TIMEOUT)
                commands.report(waypoint_done=message['request'])
                last_write = asyncio.get_running_loop().time()
            except Exception as error:
                commands.report(waypoint_done=message['request'], waypoint_error=str(error))
                raise
        try:
            await asyncio.wait_for(commands.changed.wait(), max(
                .001, HEARTBEAT_SECONDS - (asyncio.get_running_loop().time() - last_write)))
        except asyncio.TimeoutError:
            pass


async def connect_and_control(name, commands):
    while not commands.closed.is_set():
        client = None
        try:
            commands.report(message='Scanning for ' + name)
            device = await BleakScanner.find_device_by_name(name, timeout=10)
            if device is None:
                raise RuntimeError('Robot not found')
            client = BleakClient(device, disconnected_callback=lambda _: commands.changed.set())
            await asyncio.wait_for(client.connect(), timeout=10)
            await control(client, commands)
        except Exception as error:
            commands.report(error=str(error))
        finally:
            commands.ready = False
            commands.velocity = bytes(3)
            commands.position_mode = False
            commands.report(ready=False)
            if client is not None:
                try:
                    if client.is_connected:
                        await write(client, bytes(3))
                except Exception:
                    pass  # The robot's timeout also stops it if the link is gone.
                try:
                    await asyncio.wait_for(client.disconnect(), timeout=2)
                except Exception:
                    pass
        await asyncio.sleep(.5)


async def main(name):
    commands = Commands()
    Thread(target=read_commands, args=(asyncio.get_running_loop(), commands), daemon=True).start()
    owner = asyncio.create_task(connect_and_control(name, commands))
    try:
        await commands.closed.wait()
    finally:
        owner.cancel()  # Also cancels discovery or connection setup in progress.
        try:
            await owner
        except asyncio.CancelledError:
            pass


if __name__ == '__main__':
    try:
        asyncio.run(main(sys.argv[1]))
    except KeyboardInterrupt:
        pass
