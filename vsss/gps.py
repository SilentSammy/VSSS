"""World state from VSSS Provider. Coordinates are metres; angles are degrees."""
from dataclasses import dataclass
import math
from threading import Lock, Thread
import time
from types import MappingProxyType

from ._provider_observer import ProviderObserver


@dataclass(frozen=True)
class Position:
    x: float
    y: float


@dataclass(frozen=True)
class Pose(Position):
    angle: float


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
    """One world model per process, backed by the shared local VSSS Provider.

    Missing/stale objects are never reported as current.

    Configure GPS(port=..., stale_after=...) before creating any Cars if needed.
    All later GPS() calls return that same receiver.
    The default TE2004B service uses UDP 5000 data and 5001 subscriptions.
    """
    _instance = None
    _instance_lock = Lock()

    def __new__(cls, *, port=None, stale_after=None, subscribe=None):
        with cls._instance_lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._start(5000 if port is None else port,
                                .5 if stale_after is None else stale_after,
                                True if subscribe is None else subscribe)
                cls._instance = instance
            elif ((port is not None and port != cls._instance.configured_port) or
                  (stale_after is not None and stale_after != cls._instance.stale_after) or
                  (subscribe is not None and subscribe != cls._instance.subscribe)):
                raise ValueError('GPS is already running with different settings')
            return cls._instance

    def _start(self, port, stale_after, subscribe):
        if not math.isfinite(stale_after) or stale_after <= 0:
            raise ValueError('stale_after must be positive and finite')
        self.stale_after = stale_after
        if type(subscribe) is not bool:
            raise ValueError('subscribe must be a boolean')
        self.subscribe = subscribe
        self.configured_port = port
        self._lock = Lock()
        self._received = None
        self._latest = None
        self._timing_source = None
        self._best_offset = None
        self._last_source_timestamp = None
        self._delivery_problem = None
        self._listeners = set()
        self._cars, self._balls, self._board = {}, (), False
        self._vision = dict(port=port, server=('192.168.12.1' if port == 5000 else '255.255.255.255') if subscribe else None,
                            listen_port=0 if subscribe else port)
        self._observer = ProviderObserver(topics=('udp',), vision=self._vision)
        self.port = self._observer.snapshot['vision']['local_port']
        self._subscription_port = (port if port else self.port) + 1
        self._server = ('192.168.12.1', self._subscription_port) if subscribe and port == 5000 else None
        self._unicast_received = None
        self._closed = False
        self._thread = Thread(target=self._receive, name='GPS', daemon=True)
        self._thread.start()

    def _receive(self):
        while not self._closed:
            try:
                event = self._observer.read()
                if event is None:
                    continue
                if event['type'] == 'observer_overflow':
                    raise ConnectionError(event['error'])
                if event['type'] != 'udp' or not event.get('valid'):
                    continue
                address = tuple(event['sender'])
                packet = event['packet']
                cars = {int(p['id']): Pose(float(p['x']), float(p['y']), float(p['angle']))
                        for p in packet['markers']}
                balls = tuple(Position(float(p['x']), float(p['y'])) for p in packet.get('balls', []))
                # Legacy broadcast may arrive late. Once subscribed, accept
                # only our server's stream until it has been silent for 2s.
                now = event['monotonic_ns'] / 1e9
                is_unicast = self.subscribe and address[1] == self._subscription_port
                if not is_unicast and self.subscribe and self._unicast_received is not None and now - self._unicast_received < 2:
                    continue
                board = packet['board_detected']
                server_timestamp = packet.get('timestamp')
                frame_age_ms = packet.get('frame_age_ms')
                if server_timestamp is not None:
                    server_timestamp = float(server_timestamp)
                if frame_age_ms is not None:
                    frame_age_ms = float(frame_age_ms)
                if self._timing_source != address[0]:
                    self._timing_source = address[0]
                    self._best_offset = self._last_source_timestamp = None
                excess = None
                out_of_order = False
                if server_timestamp is not None:
                    offset = now-server_timestamp
                    self._best_offset = offset if self._best_offset is None else min(self._best_offset,offset)
                    excess = max(0.,offset-self._best_offset)
                    out_of_order = (self._last_source_timestamp is not None
                                    and server_timestamp < self._last_source_timestamp)
                minimum_age = (0. if frame_age_ms is None else frame_age_ms/1000) + (excess or 0.)
                usable = not out_of_order and minimum_age <= self.stale_after
                visible_cars, visible_balls = (cars, balls) if board else ({}, ())
                feedback = Feedback(now, board, MappingProxyType(visible_cars.copy()), visible_balls,
                                    server_timestamp, packet.get('frame_sequence'), frame_age_ms,
                                    excess, minimum_age, usable)
                with self._lock:
                    if usable:
                        self._cars, self._balls = visible_cars, visible_balls
                        self._board = board
                        self._received = now
                        self._latest = feedback
                        self._last_source_timestamp = server_timestamp
                        self._delivery_problem = None
                        if is_unicast:
                            self._server = address
                            self._unicast_received = now
                    else:
                        self._delivery_problem = 'Camera packets are delayed or out of order.'
                    listeners = tuple(self._listeners)
                for listener in listeners:
                    try:
                        listener(feedback)
                    except Exception as error:
                        print(f'[GPS] Removing failed feedback listener: {error}', flush=True)
                        self.remove_listener(listener)
            except (OSError, ValueError):
                if self._closed:
                    break
                with self._lock:
                    self._received = None
                    self._delivery_problem = 'Camera provider interrupted; reconnecting.'
                self._observer.close()
                while not self._closed:
                    try:
                        observer = ProviderObserver(topics=('udp',), vision=self._vision)
                        if self._closed:
                            observer.close()
                            break
                        self._observer = observer
                        self.port = observer.snapshot['vision']['local_port']
                        self._timing_source = self._best_offset = self._last_source_timestamp = None
                        self._unicast_received = None
                        break
                    except (OSError, ValueError):
                        time.sleep(.2)

    def _snapshot(self):
        with self._lock:
            if (self._received is None or
                    self._latest.minimum_age_at_receipt_s + time.monotonic() - self._received > self.stale_after):
                return {}, (), False
            return self._cars.copy(), self._balls, self._board

    @property
    def cars(self):
        return self._snapshot()[0]

    @property
    def balls(self):
        return self._snapshot()[1]

    @property
    def board_detected(self):
        return self._snapshot()[2]

    @property
    def feedback_problem(self):
        """A useful explanation when a controller is missing position feedback."""
        with self._lock:
            if self._received is None:
                return self._delivery_problem or f'No camera telemetry on UDP {self.port}; check lab WiFi and the detection server.'
            if self._latest.minimum_age_at_receipt_s + time.monotonic() - self._received > self.stale_after:
                return self._delivery_problem or 'Camera telemetry is stale; check lab WiFi and the detection server.'
            if not self._board:
                return 'The camera cannot detect the board.'
        return None

    def get_car(self, number):
        return self.cars.get(number)

    @property
    def latest(self):
        """Last usable packet including timing metadata; may subsequently become stale."""
        with self._lock:
            return self._latest

    def add_listener(self, listener):
        """Observe every valid packet. Callbacks must return quickly (e.g. queue it)."""
        with self._lock:
            self._listeners.add(listener)

    def remove_listener(self, listener):
        with self._lock:
            self._listeners.discard(listener)

    def close(self):
        """Release the shared receiver, usually only needed by tools/tests."""
        with type(self)._instance_lock:
            self._closed = True
            self._observer.close()
            self._thread.join(timeout=1)
            if type(self)._instance is self:
                type(self)._instance = None
