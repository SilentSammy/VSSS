"""Stream recorded events from JSONL without loading an entire bag into RAM."""
import json
import math
from pathlib import Path
from threading import Event
import time


class BagReader:
    def __init__(self, path, topics=('udp', 'ble'), speed=1.):
        self.path = Path(path)
        self.topics = frozenset(topics)
        if not self.topics or not self.topics <= {'udp', 'ble'}:
            raise ValueError('topics must contain udp and/or ble')
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError('speed must be positive and finite')
        self.speed = float(speed)
        self._closed = Event()
        self._file = self.path.open(encoding='utf-8')
        self.snapshot = None
        self.complete = None
        self._line = 0
        self._started = None
        self._read_error = None
        try:
            self._pending = self._next_event()
            self._first_timestamp = self._pending[0] if self._pending else None
            if self.snapshot is None:
                self.snapshot = dict(type='provider_snapshot', controlled_cars=[], vision=None)
            self.snapshot = dict(self.snapshot, replay=dict(path=str(self.path), speed=self.speed))
        except Exception:
            self.close()
            raise

    def _next_event(self):
        for line in self._file:
            self._line += 1
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'invalid JSON on bag line {self._line}: {error.msg}') from error
            if not isinstance(event, dict):
                raise ValueError(f'bag line {self._line} must contain a JSON object')
            kind = event.get('type')
            if kind == 'bag_header':
                if not isinstance(event.get('provider'), dict):
                    raise ValueError(f'bag header on line {self._line} has no provider snapshot')
                self.snapshot = event['provider']
            elif kind == 'provider_snapshot' and self.snapshot is None:
                self.snapshot = event
            elif kind == 'session_end':
                self.complete = bool(event.get('complete'))
            if kind in self.topics or ('ble' in self.topics and kind in
                    ('control_acquired', 'control_released', 'connection', 'error')):
                timestamp = event.get('monotonic_ns', event.get('wall_time_ns'))
                if type(timestamp) not in (int, float) or not math.isfinite(timestamp):
                    raise ValueError(f'bag event on line {self._line} has no valid timestamp')
                return int(timestamp), event
        return None

    @property
    def finished(self):
        return self._closed.is_set() or self._pending is None

    @property
    def empty(self):
        return self._first_timestamp is None

    def rewind(self):
        self._file.seek(0)
        self._line = 0
        self._started = None
        self.complete = None
        self._pending = self._next_event()
        self._first_timestamp = self._pending[0] if self._pending else None

    def read(self, timeout=.2):
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError('timeout must be finite and nonnegative')
        if self._read_error is not None:
            raise self._read_error
        if self.finished:
            return None
        if self._started is None:
            self._started = time.monotonic()
        timestamp, event = self._pending
        due = self._started + (timestamp - self._first_timestamp) / 1e9 / self.speed
        remaining = due - time.monotonic()
        if remaining > timeout:
            self._closed.wait(timeout)
            return None
        if remaining > 0 and self._closed.wait(remaining):
            return None
        if self._closed.is_set():
            return None
        try:
            self._pending = self._next_event()
        except (OSError, ValueError, KeyError, TypeError) as error:
            self._read_error = error
        return event

    def close(self):
        self._closed.set()
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
