"""Interchangeable live and recorded VSSS event providers.

Providers expose the same small observation interface: ``snapshot``,
``read(timeout)``, ``finished``, ``close()``, and context-manager support.
``SimulationProvider`` can implement this contract later without changing
consumers such as the field plotter.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from threading import Event
import time

from ._provider_observer import ProviderObserver


class LiveProvider:
    """Observe the shared local provider backed by real UDP/BLE hardware."""

    def __init__(self, topics=('udp', 'ble'), vision=None):
        self._observer = ProviderObserver(topics, vision)
        self.snapshot = self._observer.snapshot

    @property
    def finished(self):
        return False

    def read(self, timeout=.2):
        return self._observer.read(timeout)

    def close(self):
        self._observer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class BagProvider:
    """Replay a vsss-observer JSONL bag using its original relative timing."""

    def __init__(self, path, topics=('udp', 'ble'), speed=1.0):
        self.path = Path(path)
        self.topics = frozenset(topics)
        if not self.topics or not self.topics <= {'udp', 'ble'}:
            raise ValueError('topics must contain udp and/or ble')
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError('speed must be positive and finite')
        self.speed = float(speed)
        self._events = []
        self.snapshot = None
        for number, line in enumerate(self.path.read_text(encoding='utf-8').splitlines(), 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'invalid JSON on bag line {number}: {error.msg}') from error
            kind = event.get('type')
            if kind == 'provider_snapshot' and self.snapshot is None:
                self.snapshot = event
            if kind in self.topics:
                timestamp = event.get('monotonic_ns', event.get('wall_time_ns'))
                if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
                    raise ValueError(f'bag event on line {number} has no valid timestamp')
                self._events.append((int(timestamp), event))
        if self.snapshot is None:
            self.snapshot = dict(type='provider_snapshot', controlled_cars=[],
                                 vision=dict(source='bag', path=str(self.path)))
        else:
            self.snapshot = dict(self.snapshot, replay=dict(path=str(self.path), speed=self.speed))
        self._index = 0
        self._closed = Event()
        self._started = None
        self._first_timestamp = self._events[0][0] if self._events else 0

    @property
    def finished(self):
        return self._closed.is_set() or self._index >= len(self._events)

    def read(self, timeout=.2):
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError('timeout must be finite and nonnegative')
        if self.finished:
            return None
        if self._started is None:
            self._started = time.monotonic()
        timestamp, event = self._events[self._index]
        due = self._started + (timestamp - self._first_timestamp) / 1e9 / self.speed
        remaining = due - time.monotonic()
        if remaining > timeout:
            self._closed.wait(timeout)
            return None
        if remaining > 0 and self._closed.wait(remaining):
            return None
        if self._closed.is_set():
            return None
        self._index += 1
        return event

    def close(self):
        self._closed.set()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
