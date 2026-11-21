"""Small kinematic simulation: body-frame commands, bounded poses, manual edits."""
from dataclasses import replace
from copy import deepcopy
import json
import math
from pathlib import Path
from threading import Event, RLock, Thread
import time

from .components import CarCmd, GPSState, rpc_method
from .events import EventHub
from .models import BallState, CarState, Field, Pose, StatePatch, Velocity, WorldState, decode
from .provider import VSSSProvider


def scenario_state(path):
    """Load a tagged WorldState or the header of a new-format recording."""
    with Path(path).open(encoding="utf-8") as file:
        value = json.loads(file.readline())
    if value.get("schema") == "vsss.recording/v1":
        value = value["state"]
    state = decode(value)
    if not isinstance(state, WorldState):
        raise ValueError("scenario must contain a WorldState or vsss.recording/v1 header")
    return state


class CarCmdSim(CarCmd):
    def __init__(self, events, simulator):
        super().__init__(events)
        self.simulator = simulator

    @rpc_method
    def available(self):
        return [car.id for car in self.simulator.gps.state().cars]

    def _apply(self, car_id, velocity):
        with self.simulator._world_lock:
            if car_id not in {car.id for car in self.simulator._world.cars}:
                raise ValueError("unknown car")
            self.simulator._velocities[car_id] = velocity


class VSSSSim(VSSSProvider):
    def __init__(self, state=None, *, rate=30):
        if not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 1 <= rate <= 200:
            raise ValueError("simulation rate must be 1..200 Hz")
        events = EventHub()
        self._world_lock = RLock()
        self._world = self._validated(state or WorldState(cars=[CarState(1)], balls=[BallState(0, Pose(.3, .3))]))
        self._velocities = {}
        self._stop = Event()
        self._thread = None
        super().__init__(GPSState(events, self._world), CarCmdSim(events, self), mode="simulation")
        self._rate = rate

    @staticmethod
    def _validated(state):
        if not isinstance(state, WorldState):
            raise TypeError("expected WorldState")
        for objects, radius in ((state.cars, .07), (state.balls, .02)):
            if state.field.width < 2 * radius or state.field.height < 2 * radius:
                raise ValueError("field is too small for object bodies")
            for obj in objects:
                if abs(obj.pose.x) > state.field.width / 2 - radius or abs(obj.pose.y) > state.field.height / 2 - radius:
                    raise ValueError("object lies outside field bounds")
        return deepcopy(state)

    @rpc_method
    def load_scenario(self, state):
        state = self._validated(state)
        with self.cars._lock, self._world_lock:
            self._velocities.clear()
            self.cars._owners.clear()
            self.cars._last.clear()
            self._world = replace(state, captured_at=time.time(), valid=True)
            self.gps._update(self._world)
        self.events.publish("lifecycle", {"kind": "scenario_loaded"})
        return self.gps.state()

    @rpc_method
    def update_state(self, patch):
        """Atomically edit listed poses. No implicit reset of unrelated objects."""
        if isinstance(patch, str):
            patch = decode(json.loads(patch))
        if not isinstance(patch, StatePatch):
            raise TypeError("expected StatePatch or its JSON representation")
        with self.cars._lock, self._world_lock:
            # Construct and validate the entire candidate before changing anything.
            current = self._world
            if any(not isinstance(c, CarState) for c in patch.cars) or any(not isinstance(b, BallState) for b in patch.balls):
                raise ValueError("invalid patch objects")
            cars = {c.id: c for c in patch.cars}
            balls = {b.id: b for b in patch.balls}
            if len(cars) != len(patch.cars) or len(balls) != len(patch.balls):
                raise ValueError("duplicate patch IDs")
            if not set(cars) <= {c.id for c in current.cars} or not set(balls) <= {b.id for b in current.balls}:
                raise ValueError("unknown patch object")
            candidate = self._validated(replace(current,
                cars=[cars.get(c.id, c) for c in current.cars], balls=[balls.get(b.id, b) for b in current.balls],
                captured_at=time.time(), sequence=current.sequence + 1))
            for car_id in cars:
                self._velocities.pop(car_id, None)
                self.cars._owners.pop(car_id, None)
            self._world = candidate
            self.gps._update(candidate)
        self.events.publish("lifecycle", {"kind": "state_edited", "patch": patch})
        return candidate

    def step(self, dt):
        if not math.isfinite(dt) or not 0 < dt <= .1:
            raise ValueError("step must be 0..0.1 seconds")
        with self._world_lock:
            cars = []
            for car in self._world.cars:
                velocity = self._velocities.get(car.id, Velocity())
                pose = car.pose
                angle = math.radians(pose.angle)
                # 0.6 m/s and 180 deg/s are illustrative, not identified dynamics.
                x = pose.x + .6 * (velocity.x * math.cos(angle) - velocity.y * math.sin(angle)) * dt
                y = pose.y + .6 * (velocity.x * math.sin(angle) + velocity.y * math.cos(angle)) * dt
                x = max(-self._world.field.width / 2 + .07, min(self._world.field.width / 2 - .07, x))
                y = max(-self._world.field.height / 2 + .07, min(self._world.field.height / 2 - .07, y))
                cars.append(CarState(car.id, Pose(x, y, (pose.angle + 180 * velocity.w * dt + 180) % 360 - 180)))
            self._world = replace(self._world, cars=cars, sequence=self._world.sequence + 1,
                                  captured_at=time.time(), valid=True)
            self.gps._update(self._world)

    def start(self):
        if self._thread:
            return self
        def run():
            last = time.monotonic()
            while not self._stop.wait(1 / self._rate):
                now = time.monotonic()
                self.step(min(now - last, .1))
                last = now
        self._thread = Thread(target=run, daemon=True)
        self._thread.start()
        return self

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        super().close()


def main(argv=None):
    from .cli import provider_parser, run_provider
    parser = provider_parser(__doc__)
    parser.add_argument("scenario", nargs="?")
    parser.add_argument("--rate", type=float, default=30)
    args = parser.parse_args(argv)
    state = scenario_state(args.scenario) if args.scenario else None
    run_provider(VSSSSim(state, rate=args.rate).start(), args)


if __name__ == "__main__":
    main()
