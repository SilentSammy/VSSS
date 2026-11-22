"""Small student view over the provider's freshness-checked GPS methods."""
from dataclasses import dataclass
from types import MappingProxyType
from .models import Pose, finite
from .provider import VSSSProvider


@dataclass(frozen=True)
class Position:
    x: float
    y: float


@dataclass(frozen=True)
class Feedback:
    received_at: float
    board_detected: bool
    cars: object
    balls: tuple
    server_timestamp: object = None
    frame_sequence: object = None
    frame_age_ms: object = None
    network_excess_delay_s: object = None
    minimum_age_at_receipt_s: float = 0.
    usable: bool = True


class GPS:
    def __init__(self, *, endpoint=None, provider=None, stale_after=.5):
        finite(stale_after)
        if stale_after <= 0: raise ValueError('stale_after must be positive')
        self.stale_after = stale_after
        self._provider = provider or VSSSProvider.connect(endpoint)
        self._owns_provider = provider is None
        self._listeners, self._subscription = set(), None
        self._closed = False

    def _state(self):
        return self._provider.gps.state(stale_after=self.stale_after)

    @property
    def cars(self):
        state = self._state()
        return {c.id: c.pose for c in state.cars} if state and state.valid else {}

    @property
    def balls(self):
        state = self._state()
        return tuple(Position(b.pose.x, b.pose.y) for b in state.balls) if state and state.valid else ()

    @property
    def ball(self):
        """The first visible ball's position, or None when no ball is visible."""
        balls = self.balls
        return balls[0] if balls else None

    @property
    def board_detected(self):
        state = self._state()
        return bool(state and state.valid)

    @property
    def feedback_problem(self):
        state = self._state()
        return self._provider.gps.problem() or (None if state and state.valid else 'Camera feedback is stale')

    def get_car(self, number):
        return self.cars.get(number)

    @staticmethod
    def _feedback(state):
        return Feedback(state.received_at, state.valid,
            MappingProxyType({c.id: c.pose for c in state.cars} if state.valid else {}),
            tuple(Position(b.pose.x, b.pose.y) for b in state.balls) if state.valid else (),
            state.captured_at, state.sequence, state.frame_age_ms,
            max(0., state.source_age-state.frame_age_ms/1000), state.source_age, state.valid)

    @property
    def latest(self):
        state = self._state()
        return self._feedback(state) if state else None

    def add_listener(self, listener):
        self._listeners.add(listener)
        if self._subscription is None:
            self._subscription = self._provider.subscribe(['gps'], self._changed)

    def _changed(self, event):
        feedback = self._feedback(event.value)
        for listener in tuple(self._listeners): listener(feedback)

    def remove_listener(self, listener): self._listeners.discard(listener)

    def close(self):
        if self._closed: return
        self._closed = True
        if self._subscription: self._subscription.close()
        if self._owns_provider: self._provider.close()


class _LazyGPS:
    def __init__(self): self._instance = None
    def __getattr__(self, name):
        if self._instance is None: self._instance = GPS()
        return getattr(self._instance, name)
    def close(self):
        if self._instance is not None:
            self._instance.close()
            self._instance = None


gps = _LazyGPS()
