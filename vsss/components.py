"""Ordinary component methods; RPC is only an explicit exposure marker."""
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from contextvars import ContextVar
from dataclasses import replace
from threading import RLock
import time
from .models import Velocity, Pose, WorldState, CommandEvent, finite

session_context = ContextVar('vsss_session', default='local')


def rpc_method(method):
    method.rpc_exposed = True
    return method


class GPS(ABC):
    def __init__(self, events): self.events = events
    @rpc_method
    @abstractmethod
    def state(self, stale_after=None): """Current usable world, with no stale/missing objects."""
    def close(self): pass


class GPSState(GPS):
    def __init__(self, events, initial=None, *, stale_after=.5):
        super().__init__(events)
        finite(stale_after)
        if stale_after <= 0: raise ValueError('stale_after must be positive')
        self.stale_after = stale_after
        self._state = initial or WorldState(valid=False)
        self._lock, self._received = RLock(), None
        self._problem = None

    @rpc_method
    def snapshot(self):
        """Last observation, including historical/invalid objects."""
        with self._lock: return deepcopy(self._state)

    @rpc_method
    def state(self, stale_after=None):
        limit = self.stale_after if stale_after is None else min(self.stale_after, stale_after)
        finite(limit)
        if limit <= 0: raise ValueError('stale_after must be positive')
        with self._lock:
            state, age = deepcopy(self._state), self.age()
            if not state.valid or (age if age is not None else state.source_age) > limit:
                state = replace(state, valid=False, cars=[], balls=[])
            return state

    @rpc_method
    def get_car(self, number):
        return next((car.pose for car in self.state().cars if car.id == number), None)

    @rpc_method
    def problem(self):
        with self._lock:
            if self._problem: return self._problem
            if not self._state.valid: return 'Board is not detected'
            age = self.age()
            return 'Camera feedback is stale' if age is not None and age > self.stale_after else None

    def _update(self, state):
        if not isinstance(state, WorldState): raise TypeError('expected WorldState')
        with self._lock:
            self._state = deepcopy(state)
            self._received = time.monotonic()
            self._problem = None
        self.events.publish('gps', state)

    @rpc_method
    def age(self):
        with self._lock:
            return None if self._received is None else self._state.source_age + time.monotonic() - self._received


class CarCmd(ABC):
    def __init__(self, events):
        self.events = events
        self._lock = RLock()  # Metadata only; never held across device I/O.
        self._car_locks, self._owners, self._last = {}, {}, {}
        self._statuses = {}

    def _car_lock(self, car_id):
        if type(car_id) is not int or car_id < 0: raise ValueError('expected nonnegative car ID')
        with self._lock: return self._car_locks.setdefault(car_id, RLock())

    @rpc_method
    @abstractmethod
    def available(self, refresh=False): """Known car IDs."""
    @abstractmethod
    def _apply(self, car_id, velocity): """Apply and acknowledge one command."""
    def _waypoint(self, car_id, target): raise NotImplementedError('waypoints unavailable')
    def _operation(self, car_id, operation, value): raise NotImplementedError(operation)

    def _command(self, car_id, operation, value):
        with self._car_lock(car_id):
            (self._apply if operation == 'velocity' else self._waypoint)(car_id, value)
            owner = session_context.get()
            event = CommandEvent(car_id, operation, value, owner)
            with self._lock:
                if operation == 'velocity' and value == Velocity(): self._owners.pop(car_id, None)
                else: self._owners[car_id] = owner
                self._last[car_id] = event
            self.events.publish('commands', event)
            return event

    @rpc_method
    def set_velocity(self, car_id, velocity):
        if not isinstance(velocity, Velocity): raise ValueError('expected Velocity')
        return self._command(car_id, 'velocity', velocity)
    @rpc_method
    def go_to(self, car_id, target):
        if not isinstance(target, Pose): raise ValueError('expected Pose waypoint')
        return self._command(car_id, 'waypoint', target)
    @rpc_method
    def stop(self, car_id): return self.set_velocity(car_id, Velocity())
    @rpc_method
    def status(self, car_id): return self._settings(car_id, 'status')
    @rpc_method
    def connection(self, car_id): return self._settings(car_id, 'connection')
    @rpc_method
    def connect(self, car_id): return self._settings(car_id, 'connect')
    @rpc_method
    def release(self, car_id):
        """Release this session's motion without disturbing a newer owner."""
        self._stop_owned(car_id, session_context.get())
        return True

    def _quiet_stop(self, car_id): self._apply(car_id, Velocity())

    def _settings(self, car_id, operation, value=None):
        with self._car_lock(car_id):
            if operation.startswith('set_') or operation in ('restart', 'reconnect_telemetry'):
                self._quiet_stop(car_id)
                with self._lock: self._owners.pop(car_id, None)
                self._observe(CommandEvent(car_id, 'velocity', Velocity(), session_context.get(), operation))
            result = self._operation(car_id, operation, value)
            if operation == 'status':
                with self._lock:
                    changed = self._statuses.get(car_id) != result
                    self._statuses[car_id] = result
                if changed:
                    self.events.publish('lifecycle', dict(kind='car_status', car_id=car_id, status=result))
            # Arguments/results may contain credentials: never put them in events.
            if operation.startswith('set_') or operation in ('restart', 'reconnect_telemetry'):
                self.events.publish('lifecycle', dict(kind='car_operation', car_id=car_id, operation=operation))
            return result

    @rpc_method
    def read_trim(self, car_id): return self._settings(car_id, 'read_trim')
    @rpc_method
    def set_trim(self, car_id, matrix=None):
        from ._ble_protocol import normalize_trim_matrix
        return self._settings(car_id, 'set_trim', normalize_trim_matrix(matrix))
    @rpc_method
    def read_controller(self, car_id): return self._settings(car_id, 'read_controller')
    @rpc_method
    def set_controller(self, car_id, parameters): return self._settings(car_id, 'set_controller', parameters)
    @rpc_method
    def read_robot_config(self, car_id): return self._settings(car_id, 'read_robot_config')
    @rpc_method
    def set_robot_config(self, car_id, configuration):
        from ._ble_protocol import normalize_robot_config
        return self._settings(car_id, 'set_robot_config', normalize_robot_config(configuration))
    @rpc_method
    def reconnect_telemetry(self, car_id): return self._settings(car_id, 'reconnect_telemetry')
    @rpc_method
    def restart(self, car_id): return self._settings(car_id, 'restart')
    @rpc_method
    def observed(self):
        with self._lock: return deepcopy(list(self._last.values()))

    @rpc_method
    def statuses(self):
        """Last observed onboard statuses; this snapshot never performs I/O."""
        with self._lock:
            return [dict(car_id=number, status=deepcopy(status)) for number, status in self._statuses.items()]

    def _observe(self, event):
        if not isinstance(event, CommandEvent): raise TypeError('expected CommandEvent')
        with self._lock: self._last[event.car_id] = deepcopy(event)
        self.events.publish('commands', event)

    def _stop_owned(self, car_id, owner):
        with self._car_lock(car_id):
            with self._lock:
                if self._owners.get(car_id) != owner: return
            self._apply(car_id, Velocity())
            with self._lock: self._owners.pop(car_id, None)
            self._observe(CommandEvent(car_id, 'velocity', Velocity(), owner, 'disconnect'))

    def disconnect(self, session):
        with self._lock:
            cars = [car_id for car_id, owner in self._owners.items() if owner == session]
        if cars:
            with ThreadPoolExecutor(max_workers=len(cars)) as pool:
                results = [pool.submit(self._stop_owned, car_id, session) for car_id in cars]
                errors = []
                for result in results:
                    try: result.result()
                    except Exception as error: errors.append(error)
                if errors: raise errors[0]

    def close(self):
        with self._lock: sessions = set(self._owners.values())
        for session in sessions: self.disconnect(session)
