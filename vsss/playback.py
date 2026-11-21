"""Play domain recordings through the same GPS and command observation API."""
import math
from threading import Event, Thread
import time

from .components import CarCmd, GPSState, rpc_method
from .events import EventHub
from .models import CommandEvent, WorldState
from .provider import VSSSProvider
from .recording import read_header, read_events


class GPSPlayback(GPSState):
    def _receive(self, state):
        self._update(state)


class CarCmdPlayback(CarCmd):
    def __init__(self, events, gps):
        super().__init__(events)
        self.gps = gps

    @rpc_method
    def available(self):
        return sorted({car.id for car in self.gps.state().cars} | set(self._last))

    def _apply(self, car_id, velocity):
        raise RuntimeError("playback is read-only")

    def _receive(self, command):
        self._observe(command)


class VSSSPlayback(VSSSProvider):
    def __init__(self, path, *, speed=1., wait_for_subscriber=True, loop=False):
        if not isinstance(speed, (int, float)) or not math.isfinite(speed) or speed <= 0:
            raise ValueError("playback speed must be positive and finite")
        self.initial, self.initial_commands = read_header(path)
        events = EventHub()
        gps = GPSPlayback(events, self.initial)
        cars = CarCmdPlayback(events, gps)
        for command in self.initial_commands:
            cars._last[command.car_id] = command
        super().__init__(gps, cars, mode="playback")
        self.path, self.speed, self.loop = path, speed, loop
        self.finished = Event()
        self.error = None
        self._stop, self._start = Event(), Event()
        self._thread = None
        events.on_subscribe = self._start.set
        if not wait_for_subscriber:
            self._start.set()

    @rpc_method
    def start_playback(self):
        self._start.set()

    def start(self):
        if self._thread:
            return self
        self._thread = Thread(target=self._play, daemon=True)
        self._thread.start()
        return self

    def _play(self):
        try:
            while not self._stop.is_set() and not self._start.wait(.1):
                pass
            while not self._stop.is_set():
                self.gps._receive(self.initial)
                with self.cars._lock:
                    self.cars._last.clear()
                for command in self.initial_commands:
                    self.cars._receive(command)
                start = time.monotonic()
                for offset, event in read_events(self.path):
                    if self._stop.wait(max(0., start + offset / self.speed - time.monotonic())):
                        return
                    if event.topic == "gps" and isinstance(event.value, WorldState):
                        self.gps._receive(event.value)
                    elif event.topic == "commands" and isinstance(event.value, CommandEvent):
                        self.cars._receive(event.value)
                    elif event.topic == "lifecycle":
                        self.events.publish("lifecycle", event.value)
                    else:
                        raise ValueError("event topic/value mismatch")
                if not self.loop:
                    self.events.publish("lifecycle", {"kind": "playback_end", "complete": True})
                    return
                if self._stop.wait(.01):
                    return
        except Exception as error:
            self.error = error
            self.events.publish("lifecycle", {"kind": "playback_end", "complete": False, "error": str(error)})
        finally:
            self.finished.set()

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        super().close()


def main(argv=None):
    from .cli import provider_parser, run_provider
    parser = provider_parser(__doc__)
    parser.add_argument("recording")
    parser.add_argument("--speed", type=float, default=1.)
    parser.add_argument("--immediate", action="store_true")
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args(argv)
    run_provider(VSSSPlayback(args.recording, speed=args.speed,
                             wait_for_subscriber=not args.immediate, loop=args.loop).start(), args)


if __name__ == "__main__":
    main()
