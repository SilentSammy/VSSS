"""Version-one domain recordings: world state and command events, no UDP/BLE."""
import json
from pathlib import Path
from threading import Lock
import time

from .models import WorldState, CommandEvent, Event, encode, decode

SCHEMA = "vsss.recording/v1"


class Recorder:
    def __init__(self, client, path):
        # Refuse to overwrite an experiment by accident.
        self._file = Path(path).open("x", encoding="utf-8")
        self._lock = Lock()
        self._start = time.monotonic()
        self._subscription = None
        try:
            # Holding this lock keeps event callbacks behind the header write.
            with self._lock:
                self._subscription = client.subscribe(["gps", "commands", "lifecycle"], self._event, capacity=4096)
                self._write({"schema": SCHEMA, "state": encode(client.gps.state()),
                             "commands": encode(client.cars.observed()),
                             "statuses": encode(client.cars.statuses()), "created_at": time.time()})
        except Exception:
            if self._subscription:
                self._subscription.close()
            self._file.close()
            raise

    def _write(self, value):
        self._file.write(json.dumps(value, allow_nan=False) + "\n")
        self._file.flush()

    def _event(self, event):
        with self._lock:
            self._write({"type": "event", "time": time.monotonic() - self._start, "event": encode(event)})

    def close(self):
        if self._subscription:
            self._subscription.close()
        error = self._subscription.error if self._subscription else None
        with self._lock:
            if not self._file.closed:
                self._write({"type": "end", "complete": error is None})
                self._file.close()
        if error:
            raise RuntimeError("recording incomplete") from error

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def read_header(path, *, with_status=False):
    with Path(path).open(encoding="utf-8") as file:
        header = json.loads(file.readline())
    if header.get("schema") != SCHEMA:
        raise ValueError(f"expected {SCHEMA}; legacy protocol bags are intentionally unsupported")
    state = decode(header["state"])
    commands = decode(header.get("commands", []))
    if not isinstance(state, WorldState) or any(not isinstance(c, CommandEvent) for c in commands):
        raise ValueError("invalid recording header")
    if with_status:
        from ._ble_protocol import RobotStatus
        statuses = decode(header.get('statuses', []))
        if not isinstance(statuses, list) or any(not isinstance(item, dict) or type(item.get('car_id')) is not int
                or item['car_id'] < 0 or not isinstance(item.get('status'), RobotStatus) for item in statuses):
            raise ValueError('invalid recording statuses')
        return state, commands, statuses
    return state, commands


def read_events(path):
    with Path(path).open(encoding="utf-8") as file:
        next(file)
        previous = 0.
        for line in file:
            entry = json.loads(line)
            if entry.get("type") == "end":
                if entry.get("complete") is not True:
                    raise ValueError("recording marked incomplete")
                if any(tail.strip() for tail in file):
                    raise ValueError("data follows recording end")
                return
            offset = entry["time"]
            from .models import finite
            finite(offset)
            if entry.get("type") != "event" or offset < previous:
                raise ValueError("invalid recording event order")
            event = decode(entry["event"])
            if not isinstance(event, Event):
                raise ValueError("invalid event")
            previous = offset
            yield offset, event
    raise ValueError("recording has no clean end marker")
