"""Simulation served through the same authenticated provider as live hardware."""
import base64
from collections import deque
from contextlib import nullcontext
import json
import struct
from threading import Event, RLock, Thread, current_thread
import time
import uuid

from mp.onboard_controller import OnboardController
from . import _ble_protocol as protocol
from ._ble_client import normalize_trim_matrix
from ._provider import Provider, stamp
from ._provider_car import CarSession
from ._sim import SimWorld


class SimCarSession(CarSession):
    """Reuse live session fanout, ownership, and departure-stop rules, without BLE."""
    def __init__(self, name, emit, provider, car):
        self.emit, self.provider, self.car = emit, provider, car
        self.lock = RLock()
        self.clients, self.pending, self.velocities = {}, {}, {}
        self.owner = None
        self.closed = False
        self.next_sequence = 0
        self.trace_sequence = 0
        self.connection_id = uuid.uuid4().hex
        self.last_heartbeat = time.monotonic()
        self.cache = {}
        self.commands = deque()
        self.changed = Event()
        with provider.world_lock:
            car.stop()
        self.emit('connection', ready=True, simulated=True)
        self.thread = Thread(target=self.process_commands, name=f'sim-{name}', daemon=True)
        self.thread.start()

    def attach(self):
        with self.lock:
            with self.provider.world_lock:
                self.cache = dict(ready=True, trim_matrix=list(self.car.trim),
                                  controller_config=dict(self.car.controller),
                                  robot_status=self.provider.world.status(self.car))
            return super().attach()

    def trace(self, characteristic, operation, payload, session_id, reason='command'):
        uuids = dict(velocity=protocol.VELOCITY_UUID, lease=protocol.LEASE_UUID,
                     trim=protocol.TRIM_MATRIX_UUID, controller=protocol.CONTROLLER_UUID,
                     waypoint=protocol.WAYPOINT_UUID, status=protocol.STATUS_UUID,
                     recovery=protocol.CONTROL_UUID)
        self.trace_sequence += 1
        common = dict(operation=operation, characteristic=characteristic, uuid=uuids[characteristic],
                      request=self.trace_sequence, connection_id=self.connection_id,
                      origin_session_id=session_id, simulated=True, reason=reason)
        for phase in ('start', 'complete'):
            details = dict(common, phase=phase)
            if (operation == 'write' and phase == 'start') or (operation == 'read' and phase == 'complete'):
                details['payload_hex'] = payload.hex()
            self.emit('ble_event', event=stamp('ble', **details), session_id=session_id)

    def broadcast(self, message):
        for client in self.clients:
            self.deliver(client, message)

    def write(self, message):
        # Also used by the shared detach implementation for owner departure.
        values = message.get('velocity')
        if not isinstance(values, list) or len(values) != 3 or any(type(v) is not int or not 0 <= v <= 255 for v in values):
            raise ValueError('velocity requires three bytes')
        command = tuple(v / 128 for v in struct.unpack('<bbb', bytes(values)))
        with self.provider.world_lock:
            self.car.navigator.clear_target()
            self.car.fault = 0
            if not any(command):
                self.car.stop()
            self.car.command = command
            self.car.mode = 1 if any(command) else 0
        self.trace('velocity', 'write', bytes(values), message.get('_session_id'),
                   'command' if self.owner else 'owner_departure_stop')

    def command(self, session_id, message):
        if not isinstance(message, dict) or not any(key in message for key in ('velocity', 'waypoint', 'operation')):
            raise ValueError('Unknown car command')
        with self.lock:
            if self.closed:
                raise ConnectionError('Simulated car session has ended')
            if len(self.commands) >= 4096:
                raise ValueError('Simulated command queue full')
            # Model transit time, not a blocking sleep in the connection reader.
            self.commands.append((time.perf_counter() + self.provider.world.ble_latency, session_id, dict(message)))
            self.changed.set()

    def process_commands(self):
        while True:
            with self.lock:
                if self.closed:
                    return
                delay = None
                if self.commands:
                    delay = max(0., self.commands[0][0] - time.perf_counter())
                    if delay == 0:
                        _, client, message = self.commands.popleft()
                        if client in self.clients:
                            try:
                                self.apply_command(client, message)
                            except (ValueError, TypeError, KeyError, OverflowError) as error:
                                self.deliver(client, dict(ready=False, error=str(error)))
                                self.deliver(client, None)
                        continue
                self.changed.clear()
            self.changed.wait(delay)

    def cancel_pending_motion(self):
        self.commands = deque(item for item in self.commands
                              if 'velocity' not in item[2] and 'waypoint' not in item[2])
        self.changed.set()

    def detach(self, session_id):
        with self.lock:
            self.commands = deque(item for item in self.commands if item[1] != session_id)
            super().detach(session_id)
            self.changed.set()

    def apply_command(self, session_id, message):
        with self.lock:
            if 'velocity' in message:
                # Validate before allowing a malformed command to take ownership.
                values = message['velocity']
                if not isinstance(values, list) or len(values) != 3 or any(
                        type(v) is not int or not 0 <= v <= 255 for v in values):
                    raise ValueError('velocity requires three bytes')
                self.set_owner(session_id, 'velocity')
                self.write(dict(message, _session_id=session_id))
                with self.provider.world_lock:
                    self.provider.world.waiting_for_control = False
                self.broadcast(dict(motion_changed=True))
                self.deliver(session_id, dict(written=message['sequence']))
            elif 'waypoint' in message:
                try:
                    payload = protocol.WAYPOINT_STRUCT.pack(*message['waypoint'])
                    with self.provider.world_lock:
                        self.provider.world.waypoint(self.car, message['waypoint'])
                        self.provider.world.waiting_for_control = False
                    self.set_owner(session_id, 'waypoint')
                    self.broadcast(dict(motion_changed=True))
                    self.trace('waypoint', 'write', payload, session_id)
                    self.deliver(session_id, dict(waypoint_done=message['request']))
                except (ValueError, TypeError, OverflowError, struct.error) as error:
                    self.deliver(session_id, dict(waypoint_done=message['request'], waypoint_error=str(error)))
            elif 'operation' in message:
                try:
                    with self.provider.world_lock:
                        result = self.operation(message, session_id)
                    self.cache.update(result)
                    self.broadcast(result)
                    self.deliver(session_id, dict(operation_done=message['request'], **result))
                except (ValueError, TypeError, KeyError, OverflowError, struct.error) as error:
                    self.deliver(session_id, dict(operation_done=message['request'], operation_error=str(error)))
            else:
                raise ValueError('Unknown car command')

    def operation(self, message, session_id):
        operation = message['operation']
        car = self.car
        if operation in ('read_trim', 'write_trim'):
            if operation == 'write_trim':
                values = message['values']
                if len(values) != 9:
                    raise ValueError('trim requires nine values')
                rows = normalize_trim_matrix([values[i:i + 3] for i in (0, 3, 6)])
                car.trim = tuple(v for row in rows for v in row)
                self.trace('trim', 'write', protocol.TRIM_STRUCT.pack(*car.trim), session_id)
            self.trace('trim', 'read', protocol.TRIM_STRUCT.pack(*car.trim), session_id)
            return dict(trim_matrix=list(car.trim))
        if operation in ('read_controller', 'write_controller'):
            if operation == 'write_controller':
                values = message['values']
                if len(values) != len(protocol.CONTROLLER_KEYS):
                    raise ValueError('controller requires fifteen values')
                settings = protocol.normalize_controller(dict(zip(protocol.CONTROLLER_KEYS, values)))
                payload = protocol.CONTROLLER_STRUCT.pack(*values)
                target = car.navigator.target
                car.navigator = OnboardController(**settings)
                if target is not None:
                    car.navigator.set_target(*target)
                car.controller = settings
                self.trace('controller', 'write', payload, session_id)
            payload = protocol.CONTROLLER_STRUCT.pack(*(car.controller[k] for k in protocol.CONTROLLER_KEYS))
            self.trace('controller', 'read', payload, session_id)
            return dict(controller_config=dict(car.controller))
        if operation == 'read_status':
            values = self.provider.world.status(car)
            self.trace('status', 'read', protocol.STATUS_STRUCT.pack(*values), session_id)
            return dict(robot_status=values)
        if operation in ('restart', 'reconnect_telemetry'):
            car.stop()
            self.set_owner(None, operation)
            self.broadcast(dict(motion_changed=True))
            self.trace('recovery', 'write', bytes([protocol.CONTROL_RESTART if operation == 'restart'
                                                  else protocol.CONTROL_RECONNECT_TELEMETRY]), session_id)
            return dict(robot_status=self.provider.world.status(car))
        raise ValueError('Unknown simulated car operation')

    def heartbeat(self):
        if not self.lock.acquire(blocking=False):
            return
        try:
            now = time.monotonic()
            if not self.closed and now - self.last_heartbeat >= .2:
                self.last_heartbeat = now
                self.trace('lease', 'write', b'\x01', self.owner, 'heartbeat')
        finally:
            self.lock.release()

    def close(self):
        with self.lock:
            self.closed = True
            self.commands.clear()
            self.changed.set()
            with self.provider.world_lock:
                self.car.stop()
            self.emit('connection', ready=False, simulated=True)
        if self.thread is not current_thread():
            self.thread.join(timeout=1)


class SimSource:
    def __init__(self, world):
        self.users = 0
        self.metadata_ready = Event()
        self.metadata_ready.set()
        self.info = dict(source_id=uuid.uuid4().hex, local_port=5000, port=5000,
                         subscription_port=5001, server='127.0.0.1', simulated=True,
                         metadata=dict(status=dict(setup=world.setup(), simulation=True)))

    def snapshot(self):
        return dict(self.info)


class SimProvider(Provider):
    def __init__(self, config):
        super().__init__()
        self.world = SimWorld(config)
        self.world_lock = RLock()
        self.source = SimSource(self.world)
        self.source.info['simulation_control'] = dict(source_id=self.source.info['source_id'])
        self.selected = ('sim', self.source.info['source_id'])
        self.closed = Event()
        self.thread = None
        self.run_error = None

    def should_exit(self):
        return self.closed.is_set()

    def available_cars(self, refresh=False):
        return sorted(self.world.cars)

    def acquire_source(self, hello):
        return 'sim', self.source

    def release_source(self, key, source):
        source.users -= 1

    def create_car_session(self, name, emit):
        cars = {f'car{number}': car for number, car in self.world.cars.items()}
        if name not in cars:
            raise ValueError(f'No simulated car named {name}; available IDs: {self.available_cars()}')
        return SimCarSession(name, emit, self, cars[name])

    def edit(self, editor, message):
        # Same lock ordering as normal car requests: session, then world.
        with self.car_lock:
            session = None
            if isinstance(message, dict) and message.get('kind') == 'car':
                session = self.controllers.get(f"car{message.get('id')}")
            with session.lock if session else nullcontext():
                with self.world_lock:
                    result = self.world.edit(editor, message)
                if session:
                    session.cancel_pending_motion()
                    session.set_owner(None, 'simulation_edit')
                    session.broadcast(dict(motion_changed=True))
        if message['operation'] != 'state':
            self.publish(stamp('simulation_edit', editor=editor, simulated=True,
                               operation=message['operation'], state=result))
        return result

    def simulation_control(self, conn, reader, hello):
        source_id = self.source.info['source_id']
        if hello.get('source_id') not in (None, source_id):
            conn.sendall(b'{"ready":false,"error":"Simulation instance changed"}\n')
            return
        editor = uuid.uuid4().hex
        conn.sendall((json.dumps(dict(ready=True, source_id=source_id)) + '\n').encode())
        try:
            while True:
                line = reader.readline(65537)
                if not line:
                    break
                if len(line) > 65536:
                    raise ValueError('simulation edit request too large')
                try:
                    result = self.edit(editor, json.loads(line))
                    reply = dict(ok=True, state=result)
                except (ValueError, TypeError) as error:
                    reply = dict(ok=False, error=str(error))
                conn.sendall((json.dumps(reply, allow_nan=False) + '\n').encode())
        finally:
            # A killed plot/editor must never leave objects frozen.
            with self.car_lock:
                with self.world_lock:
                    released = self.world.release_editor(editor)
                for state in released:
                    if state['kind'] == 'car':
                        session = self.controllers.get(f"car{state['id']}")
                        if session:
                            with session.lock:
                                session.cancel_pending_motion()
                                session.set_owner(None, 'simulation_editor_left')
                                session.broadcast(dict(motion_changed=True))
            for state in released:
                self.publish(stamp('simulation_edit', editor=editor, simulated=True,
                                   operation='release', state=state))

    def on_listening(self):
        self.thread = Thread(target=self.simulate, name='simulation', daemon=True)
        self.thread.start()

    def simulate(self):
        try:
            step = 1 / self.world.physics_hz
            previous = time.monotonic()
            accumulated = 0.
            next_frame = 0.
            pending = deque()
            while not self.closed.is_set():
                now = time.monotonic()
                # Never apply one large physics jump after a suspended process.
                accumulated += min(.1, now - previous)
                previous = now
                while accumulated >= step and not self.closed.is_set():
                    with self.world_lock:
                        self.world.step(step)
                        packet = None
                        if self.world.time >= next_frame:
                            packet = self.world.packet(time.monotonic())
                            next_frame += 1 / self.world.telemetry_hz
                    if packet is not None:
                        pending.append((packet['timestamp'] + self.world.udp_latency, packet))
                    accumulated -= step
                while pending and pending[0][0] <= time.monotonic():
                    _, packet = pending.popleft()
                    packet['frame_age_ms'] = max(0., (time.monotonic() - packet['timestamp']) * 1000)
                    payload = json.dumps(packet, separators=(',', ':')).encode()
                    self.publish(stamp('udp', valid=True, packet=packet, simulated=True,
                                       source_id=self.source.info['source_id'],
                                       sender=['127.0.0.1', 5001],
                                       payload_base64=base64.b64encode(payload).decode()), self.selected)
                sessions = ()
                if self.car_lock.acquire(blocking=False):
                    try:
                        sessions = tuple(self.controllers.values())
                    finally:
                        self.car_lock.release()
                for session in sessions:
                    session.heartbeat()
                self.closed.wait(min(step, max(0., step - accumulated)))
        except Exception as error:
            self.run_error = error
            self.closed.set()

    def run(self):
        try:
            super().run()
            if self.run_error is not None:
                raise RuntimeError(f'Simulation failed: {self.run_error}') from self.run_error
        finally:
            self.closed.set()
            if self.thread is not None:
                self.thread.join(timeout=2)
