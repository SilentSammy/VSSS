"""BLE owner process. stdin carries commands; stdout carries status messages.

The provider owns stdin's other end and shares this worker across local clients.
A shutdown request or provider EOF stops motion and disconnects BLE.
"""
import asyncio
import json
import sys
import time
import uuid as uuid_module
import struct
from threading import Thread
from bleak import BleakClient, BleakScanner
from vsss._ble_protocol import (CONTROL_UUID, CONTROL_RECONNECT_TELEMETRY, CONTROL_RESTART,
                          CONTROLLER_KEYS, CONTROLLER_STRUCT, CONTROLLER_UUID,
                          CONFIG_CHUNK_BYTES, CONFIG_UUID, normalize_robot_config,
                          LEASE_UUID, STATUS_STRUCT, STATUS_UUID, TRIM_MATRIX_UUID,
                          TRIM_STRUCT, VELOCITY_UUID, WAYPOINT_STRUCT, WAYPOINT_UUID)
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
        self.operations = asyncio.Queue()
        self.waypoints = asyncio.Queue()
        self.position_mode = False
        self.motion_session = None
        self.event_session = None

    def update(self, message):
        if message.get('shutdown'):
            self.closed.set()
            return
        if self.ready:
            if 'operation' in message:
                self.operations.put_nowait(message)
                self.changed.set()
                return
            if 'waypoint' in message:
                self.motion_session = message.get('_session_id')
                self.waypoints.put_nowait(message)
                self.position_mode = True
                self.changed.set()
                return
            self.velocity = bytes(message['velocity'])
            self.motion_session = message.get('_session_id')
            while not self.waypoints.empty():
                self.report(waypoint_done=self.waypoints.get_nowait()['request'],
                            waypoint_error='Waypoint superseded by velocity command')
            self.sequence = message['sequence']
            self.force = message['force']
            self.position_mode = False
            self.changed.set()

    def report(self, **message):
        try:
            print(json.dumps(message), flush=True)
        except (BrokenPipeError, OSError):
            self.closed.set()


class TracedBLE:
    """Report actual GATT operations without changing their ordering."""
    def __init__(self, client, commands):
        self.client, self.commands = client, commands
        self.sequence = 0
        self.connection_id = uuid_module.uuid4().hex

    @property
    def is_connected(self):
        return self.client.is_connected

    async def connect(self):
        return await self.client.connect()

    async def disconnect(self):
        return await self.client.disconnect()

    def event(self, phase, operation, uuid, request, **details):
        names = {VELOCITY_UUID: 'velocity', LEASE_UUID: 'lease', TRIM_MATRIX_UUID: 'trim',
                 WAYPOINT_UUID: 'waypoint', CONTROL_UUID: 'recovery',
                 CONTROLLER_UUID: 'controller', STATUS_UUID: 'status', CONFIG_UUID: 'config'}
        self.commands.report(event=dict(type='ble', phase=phase, operation=operation,
                                       uuid=str(uuid), request=request,
                                       connection_id=self.connection_id,
                                       characteristic=names.get(uuid, 'unknown'),
                                       monotonic_ns=time.monotonic_ns(), wall_time_ns=time.time_ns(),
                                       **details))

    async def write_gatt_char(self, uuid, payload, response=True, reason=None):
        self.sequence += 1
        request = self.sequence
        # The complete configuration may contain Wi-Fi credentials.  Trace its
        # operation but never place its bytes in a diagnostic event.
        details = dict(response=response, reason=reason or 'command',
                       origin_session_id=self.commands.event_session)
        if uuid != CONFIG_UUID:
            details['payload_hex'] = bytes(payload).hex()
        try:
            if uuid == VELOCITY_UUID:
                details['value'] = dict(zip(('x', 'y', 'w'), (v / 128 for v in struct.unpack('<bbb', payload))))
            elif uuid == WAYPOINT_UUID:
                details['value'] = dict(zip(('x', 'y', 'angle'), WAYPOINT_STRUCT.unpack(payload)))
            elif uuid == TRIM_MATRIX_UUID:
                details['value'] = list(TRIM_STRUCT.unpack(payload))
            elif uuid == CONTROLLER_UUID:
                details['value'] = dict(zip(CONTROLLER_KEYS, CONTROLLER_STRUCT.unpack(payload)))
        except (ValueError, struct.error):
            pass
        self.event('start', 'write', uuid, request, **details)
        try:
            result = await self.client.write_gatt_char(uuid, payload, response=response)
        except BaseException as error:
            self.event('error', 'write', uuid, request, error=str(error) or type(error).__name__,
                       origin_session_id=details['origin_session_id'])
            raise
        self.event('complete', 'write', uuid, request, origin_session_id=details['origin_session_id'])
        return result

    async def read_gatt_char(self, uuid):
        self.sequence += 1
        request = self.sequence
        origin = self.commands.event_session
        self.event('start', 'read', uuid, request, origin_session_id=origin)
        try:
            result = await self.client.read_gatt_char(uuid)
        except BaseException as error:
            self.event('error', 'read', uuid, request, error=str(error) or type(error).__name__, origin_session_id=origin)
            raise
        details = dict(origin_session_id=origin)
        if uuid != CONFIG_UUID:
            details['payload_hex'] = bytes(result).hex()
        self.event('complete', 'read', uuid, request, **details)
        return result


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


async def write(client, velocity, reason='command'):
    await asyncio.wait_for(
        client.write_gatt_char(VELOCITY_UUID, velocity, response=True, reason=reason), WRITE_TIMEOUT)


async def control(client, commands):
    # Never resume an old movement after reconnecting.
    commands.velocity = bytes(3)
    commands.sequence = 0
    commands.force = False
    commands.position_mode = False
    await write(client, bytes(3), reason='initial_stop')
    while not commands.operations.empty():
        commands.report(operation_done=commands.operations.get_nowait()['request'],
                        operation_error='Connection lost before completing BLE operation')
    while not commands.waypoints.empty():
        commands.report(waypoint_done=commands.waypoints.get_nowait()['request'],
                        waypoint_error='Connection lost before applying waypoint')
    try:
        matrix = TRIM_STRUCT.unpack(await asyncio.wait_for(
            client.read_gatt_char(TRIM_MATRIX_UUID), WRITE_TIMEOUT))
    except Exception:
        matrix = None  # Driving is still possible; configuring trim requires matrix firmware.
    matrix_supported = matrix is not None
    commands.report(trim_matrix=matrix)
    try:
        values = CONTROLLER_STRUCT.unpack(await asyncio.wait_for(
            client.read_gatt_char(CONTROLLER_UUID), WRITE_TIMEOUT))
        controller_supported = True
        commands.report(controller_config=dict(zip(CONTROLLER_KEYS, values)))
    except Exception:
        controller_supported = False
        commands.report(controller_config=None)
    # Do not read the full configuration at connection time: it may contain
    # Wi-Fi credentials and is intentionally available only on explicit API
    # calls.  Older firmware is detected when such a call is made.
    config_supported = None
    commands.ready = True
    commands.report(ready=True, message='Connected')
    last_velocity = bytes(3)
    last_sequence = 0
    last_write = asyncio.get_running_loop().time()
    while client.is_connected and not commands.closed.is_set():
        commands.changed.clear()
        velocity, sequence = commands.velocity, commands.sequence
        elapsed = asyncio.get_running_loop().time() - last_write
        commands.event_session = commands.motion_session
        if commands.position_mode:
            if elapsed >= HEARTBEAT_SECONDS:
                await asyncio.wait_for(
                    client.write_gatt_char(LEASE_UUID, bytes(1), response=True, reason='heartbeat'), WRITE_TIMEOUT)
                last_write = asyncio.get_running_loop().time()
        elif (velocity != last_velocity or elapsed >= HEARTBEAT_SECONDS
              or (commands.force and sequence != last_sequence)):
            reason = ('command' if velocity != last_velocity else
                      'forced' if commands.force and sequence != last_sequence else 'heartbeat')
            await write(client, velocity, reason=reason)
            last_velocity, last_sequence = velocity, sequence
            last_write = asyncio.get_running_loop().time()
            commands.report(written=sequence)
        if not commands.operations.empty():
            message = commands.operations.get_nowait()
            commands.event_session = message.get('_session_id')
            result = {}
            try:
                operation = message['operation']
                if operation in ('read_trim', 'write_trim'):
                    if not matrix_supported:
                        raise RuntimeError('Firmware does not support matrix trim')
                    if operation == 'write_trim':
                        payload = TRIM_STRUCT.pack(*message['values'])
                        await asyncio.wait_for(client.write_gatt_char(
                            TRIM_MATRIX_UUID, payload, response=True), WRITE_TIMEOUT)
                        last_write = asyncio.get_running_loop().time()
                    actual = await asyncio.wait_for(
                        client.read_gatt_char(TRIM_MATRIX_UUID), WRITE_TIMEOUT)
                    if operation == 'write_trim' and bytes(actual) != payload:
                        raise ValueError('Firmware rejected trim')
                    result['trim_matrix'] = TRIM_STRUCT.unpack(actual)
                elif operation in ('read_controller', 'write_controller'):
                    if not controller_supported:
                        raise RuntimeError('Firmware does not support controller configuration')
                    if operation == 'write_controller':
                        payload = CONTROLLER_STRUCT.pack(*message['values'])
                        await asyncio.wait_for(client.write_gatt_char(
                            CONTROLLER_UUID, payload, response=True), WRITE_TIMEOUT)
                        last_write = asyncio.get_running_loop().time()
                    actual = await asyncio.wait_for(
                        client.read_gatt_char(CONTROLLER_UUID), WRITE_TIMEOUT)
                    if operation == 'write_controller' and bytes(actual) != payload:
                        raise ValueError('Firmware rejected controller configuration')
                    values = CONTROLLER_STRUCT.unpack(actual)
                    result['controller_config'] = dict(zip(CONTROLLER_KEYS, values))
                elif operation == 'read_status':
                    actual = await asyncio.wait_for(
                        client.read_gatt_char(STATUS_UUID), WRITE_TIMEOUT)
                    result['robot_status'] = STATUS_STRUCT.unpack(actual)
                elif operation in ('read_robot_config', 'write_robot_config'):
                    if config_supported is False:
                        raise RuntimeError('Firmware does not support full robot configuration')
                    if operation == 'write_robot_config':
                        config = normalize_robot_config(message['config'])
                        payload = json.dumps(config, separators=(',', ':'), allow_nan=False).encode()
                        for offset in range(0, len(payload), CONFIG_CHUNK_BYTES):
                            packet = b'\x01' + struct.pack('<HH', offset, len(payload)) + payload[offset:offset + CONFIG_CHUNK_BYTES]
                            await asyncio.wait_for(client.write_gatt_char(
                                CONFIG_UUID, packet, response=True), WRITE_TIMEOUT)
                        last_write = asyncio.get_running_loop().time()
                    actual = bytes(await asyncio.wait_for(
                        client.read_gatt_char(CONFIG_UUID), WRITE_TIMEOUT))
                    result['robot_config'] = normalize_robot_config(json.loads(actual.decode()))
                    config_supported = True
                elif operation in ('reconnect_telemetry', 'restart'):
                    code = (CONTROL_RECONNECT_TELEMETRY if operation == 'reconnect_telemetry'
                            else CONTROL_RESTART)
                    await asyncio.wait_for(client.write_gatt_char(
                        CONTROL_UUID, bytes((code,)), response=True), WRITE_TIMEOUT)
                    last_write = asyncio.get_running_loop().time()
                    actual = await asyncio.wait_for(client.read_gatt_char(CONTROL_UUID), WRITE_TIMEOUT)
                    if bytes(actual) != bytes((0xa5, code)):
                        raise RuntimeError('Firmware did not accept the recovery command')
                else:
                    raise ValueError('Unknown BLE operation')
                commands.report(operation_done=message['request'], **result)
            except Exception as error:
                if operation in ('read_robot_config', 'write_robot_config'):
                    config_supported = False
                commands.report(operation_done=message['request'], operation_error=str(error))
        if not commands.waypoints.empty():
            message = commands.waypoints.get_nowait()
            commands.event_session = message.get('_session_id')
            try:
                payload = WAYPOINT_STRUCT.pack(*message['waypoint'])
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
            client = TracedBLE(BleakClient(device, disconnected_callback=lambda _: commands.changed.set()), commands)
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
                        await write(client, bytes(3), reason='disconnect_stop')
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
