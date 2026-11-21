"""Ordinary Python component interfaces. No sockets, Pyro proxies or JSON here."""
from abc import ABC, abstractmethod
from copy import deepcopy
from contextvars import ContextVar
from threading import RLock
import time

from .models import Velocity, WorldState, CommandEvent

session_context = ContextVar("vsss_session", default="local")


def rpc_method(method):
    """Mark a normal method as callable through the generic RPC endpoint."""
    method.rpc_exposed = True
    return method


class GPS(ABC):
    def __init__(self, events):
        self.events = events

    @rpc_method
    @abstractmethod
    def state(self) -> WorldState:
        """A detached snapshot of field geometry and observed object poses."""

    def close(self):
        pass


class GPSState(GPS):
    """Shared state storage/publication for live, playback and simulation."""
    def __init__(self, events, initial=None):
        super().__init__(events)
        self._state = initial or WorldState(valid=False)
        self._lock = RLock()
        self._received = None

    @rpc_method
    def state(self):
        with self._lock:
            return deepcopy(self._state)

    def _update(self, state):
        if not isinstance(state, WorldState):
            raise TypeError("expected WorldState")
        with self._lock:
            self._state = deepcopy(state)
            self._received = time.monotonic()
        self.events.publish("gps", state)

    @rpc_method
    def age(self):
        """Seconds since last delivered telemetry; None if none was received."""
        with self._lock:
            return None if self._received is None else time.monotonic() - self._received


class CarCmd(ABC):
    def __init__(self, events):
        self.events = events
        self._lock = RLock()
        self._owners = {}
        self._last = {}

    @rpc_method
    @abstractmethod
    def available(self):
        """Return known car IDs."""

    @abstractmethod
    def _apply(self, car_id, velocity):
        """Apply and acknowledge one domain command, or raise."""

    @rpc_method
    def set_velocity(self, car_id, velocity):
        if type(car_id) is not int or car_id < 0 or not isinstance(velocity, Velocity):
            raise ValueError("expected car ID and Velocity")
        with self._lock:
            self._apply(car_id, velocity)
            owner = session_context.get()
            self._owners[car_id] = owner
            event = CommandEvent(car_id, "velocity", velocity, owner)
            self._last[car_id] = event
            self.events.publish("commands", event)
            return event

    @rpc_method
    def stop(self, car_id):
        return self.set_velocity(car_id, Velocity())

    @rpc_method
    def observed(self):
        with self._lock:
            return deepcopy(list(self._last.values()))

    def _observe(self, event):
        if not isinstance(event, CommandEvent):
            raise TypeError("expected CommandEvent")
        with self._lock:
            self._last[event.car_id] = deepcopy(event)
        self.events.publish("commands", event)

    def disconnect(self, session):
        with self._lock:
            for car_id, owner in tuple(self._owners.items()):
                if owner == session:
                    self._apply(car_id, Velocity())
                    self._owners.pop(car_id, None)
                    self._observe(CommandEvent(car_id, "velocity", Velocity(), session, "disconnect"))

    def close(self):
        with self._lock:
            for car_id in tuple(self._owners):
                self._apply(car_id, Velocity())
            self._owners.clear()
