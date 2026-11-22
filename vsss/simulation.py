"""Rigid-body provider with delayed feedback, onboard waypoints and editable scenarios."""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
from threading import Event, RLock, Thread
import time
import uuid

from mp.axis_trim import AxisTrim
from mp.onboard_controller import OnboardController
from mp.robot_config import DEFAULTS
from . import _ble_protocol as protocol
from .components import CarCmd, GPSState, rpc_method, session_context
from .events import EventHub
from .models import BallState, BodyMotion, CarState, Pose, SimScenario, StatePatch, Velocity, WorldState, decode
from .physics import Physics
from .provider import VSSSProvider
from .scenarios import goalie_scenario, load_scenario as read_scenario, physical_config, save_scenario


def scenario_state(path):
    return read_scenario(path).state


class CarCmdSim(CarCmd):
    def __init__(self, events, simulator):
        super().__init__(events)
        self.simulator = simulator

    @rpc_method
    def available(self, refresh=False):
        with self.simulator._world_lock: return [car.id for car in self.simulator._world.cars]

    def _apply(self, car_id, velocity):
        self.simulator._command(car_id, 'velocity', velocity)

    def _waypoint(self, car_id, target):
        self.simulator._command(car_id, 'waypoint', target)

    def _quiet_stop(self, car_id):
        with self.simulator._world_lock:
            self.simulator._known(car_id)
            self.simulator._cancel(car_id)

    def _operation(self, car_id, operation, value):
        sim = self.simulator
        with sim._world_lock:
            sim._known(car_id)
            configuration = sim._configuration[car_id]
            if operation in ('connection', 'connect'): return dict(connected=True, generation=sim._link_generations[car_id])
            if operation == 'status': return sim._status(car_id)
            if operation == 'read_trim':
                values = configuration['trim_matrix']
                return [values[i:i+3] for i in (0, 3, 6)]
            if operation == 'set_trim':
                configuration['trim_matrix'] = [v for row in value for v in row]
                return deepcopy(value)
            if operation == 'read_controller': return deepcopy(configuration['controller'])
            if operation == 'set_controller':
                configuration['controller'] = protocol.normalize_controller(value, configuration['controller'])
                return deepcopy(configuration['controller'])
            if operation == 'read_robot_config': return deepcopy(configuration)
            if operation == 'set_robot_config':
                if value['car_id'] != car_id: raise ValueError('simulated identity is set by the scenario')
                sim._configuration[car_id] = deepcopy(value)
                sim._cancel(car_id)
                return deepcopy(value)
            if operation in ('restart', 'reconnect_telemetry'):
                sim._cancel(car_id)
                if operation == 'restart': sim._link_generations[car_id] += 1
                return True
            raise ValueError('unknown car operation')


class VSSSSim(VSSSProvider):
    def __init__(self, state=None, *, rate=30, config=None, seed=None, wait_for_control=None):
        if type(rate) not in (int, float) or not math.isfinite(rate) or not 1 <= rate <= 200:
            raise ValueError('simulation rate must be 1..200 Hz')
        scenario = state if isinstance(state, SimScenario) else SimScenario(state or
            WorldState(cars=[CarState(1)], balls=[BallState(0, Pose(.3, .3))]))
        merged = physical_config(scenario.config)
        for key, value in (config or {}).items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict): merged[key].update(value)
            else: merged[key] = value
        scenario = replace(scenario, config=physical_config(merged),
                           seed=scenario.seed if seed is None else seed,
                           wait_for_control=scenario.wait_for_control if wait_for_control is None else wait_for_control)
        self._world_lock, self._stop = RLock(), Event()
        self._thread, self._rate = None, rate
        events = EventHub()
        self._setup(scenario)
        super().__init__(GPSState(events, self._world), CarCmdSim(events, self), mode='simulation')
        self._generation = 0

    def _setup(self, scenario):
        config = physical_config(scenario.config)
        world = self._validated(scenario.state, config)
        physics = Physics(world, config, scenario.seed)
        physics.initial_motion(scenario.motion)
        self.config, self.seed = config, scenario.seed
        self._world, self._physics = world, physics
        self._waiting, self._time = scenario.wait_for_control, 0.
        self._source_origin = time.time()
        self._velocities, self._pending, self._navigation = {}, {}, {}
        self._modes = {c.id: (0, 0) for c in world.cars}
        self._configuration = {}
        self._link_generations = {c.id: 1 for c in world.cars}
        for car in world.cars:
            setting = deepcopy(DEFAULTS)
            setting['car_id'] = car.id
            self._configuration[car.id] = protocol.normalize_robot_config(setting)
        self._feedback, self._feedback_at = world, 0.
        self._feedback_pending, self._deliveries = [], []
        self._feedback_enabled, self._held = True, {}

    @staticmethod
    def _validated(state, config=None):
        if not isinstance(state, WorldState): raise TypeError('expected WorldState')
        config = physical_config(config)
        for kind, objects in [('car', state.cars), ('ball', state.balls)]:
            for obj in objects:
                if kind == 'ball':
                    vertices = [(obj.pose.x-config['ball_radius'], obj.pose.y-config['ball_radius']),
                                (obj.pose.x+config['ball_radius'], obj.pose.y+config['ball_radius'])]
                else:
                    c, s = math.cos(math.radians(obj.pose.angle)), math.sin(math.radians(obj.pose.angle))
                    vertices = [(obj.pose.x+c*x-s*y, obj.pose.y+s*x+c*y) for x, y in config['body']]
                if any(abs(x) > state.field.width/2 + .001 or abs(y) > state.field.height/2 + .001 for x, y in vertices):
                    raise ValueError('object lies outside field bounds')
        return deepcopy(state)

    @contextmanager
    def _edit_lock(self):
        with self._world_lock: ids = [c.id for c in self._world.cars]
        with ExitStack() as stack:
            for number in sorted(ids): stack.enter_context(self.cars._car_lock(number))
            stack.enter_context(self._world_lock)
            yield

    def _known(self, number):
        if number not in self._configuration: raise ValueError('unknown car')

    def _cancel(self, number):
        self._pending.pop(number, None)
        self._navigation.pop(number, None)
        self._velocities[number] = Velocity()
        self._modes[number] = (0, 0)

    def _command(self, number, operation, value):
        with self._world_lock:
            self._known(number)
            if ('car', number) in self._held: raise RuntimeError('car is held by an editor')
            if operation == 'waypoint':
                self._validated(replace(self._world, cars=[CarState(number, value)], balls=[]), self.config)
            self._waiting = False
            if operation == 'velocity' and value == Velocity():
                # Cancellation clears delayed motion as well as active motion.
                self._cancel(number)
                return
            delay = self.config['latency']['ble_ms'] / 1000
            if delay:
                self._pending[number] = (self._time+delay, operation, value)
            else:
                self._activate(number, operation, value)

    def _activate(self, number, operation, value):
        self._pending.pop(number, None)
        self._navigation.pop(number, None)
        if operation == 'velocity':
            self._velocities[number], self._modes[number] = value, (1, 0)
        else:
            controller = OnboardController(**self._configuration[number]['controller'])
            controller.set_target(value.x, value.y, math.radians(value.angle))
            self._navigation[number] = dict(controller=controller, started=self._time, recovery=None, frame=None)
            self._modes[number] = (2, 0)

    def _fresh(self, number):
        return (self._feedback.valid and self._time-self._feedback_at <= self._configuration[number]['telemetry_stale_ms']/1000
                and any(car.id == number for car in self._feedback.cars))

    def _status(self, number):
        mode, fault = self._modes[number]
        fresh = self._fresh(number)
        pose = next((c.pose for c in self._feedback.cars if c.id == number), Pose()) if fresh else Pose()
        return protocol.RobotStatus(mode, 1 | (2 if fresh else 0), fault, 0,
                                    pose.x, pose.y, pose.angle, self._feedback.sequence)

    def _navigate(self):
        for number, task in list(self._navigation.items()):
            controller = task['controller']
            if not self._fresh(number):
                self._velocities[number] = Velocity()
                controller.reset()
                task['recovery'] = self._time if task['recovery'] is None else task['recovery']
                if (self._time-task['recovery'])*1000 >= controller.telemetry_recovery_ms:
                    self._modes[number] = (4, 1)
                    self._navigation.pop(number)
                continue
            task['recovery'] = None
            if (self._time-task['started'])*1000 >= controller.waypoint_timeout_ms:
                self._velocities[number], self._modes[number] = Velocity(), (4, 2)
                self._navigation.pop(number)
                continue
            if task['frame'] != self._feedback.sequence:
                task['frame'] = self._feedback.sequence
                pose = next(c.pose for c in self._feedback.cars if c.id == number)
                command = controller.update((pose.x, pose.y, pose.angle), int(self._time*1000))
                self._velocities[number] = Velocity(*command)
                if controller.reached(int(self._time*1000)):
                    self._velocities[number], self._modes[number] = Velocity(), (3, 0)
                    self._navigation.pop(number)

    @rpc_method
    def describe(self):
        result = super().describe()
        with self._world_lock:
            result['simulation'] = dict(body=deepcopy(self.config['body']), ball_radius=self.config['ball_radius'],
                                        generation=self._generation, waiting=self._waiting)
        return result

    @rpc_method
    def load_scenario(self, state):
        scenario = state if isinstance(state, SimScenario) else SimScenario(state)
        # Validate fully before touching ownership or the live world.
        physical_config(scenario.config)
        self._validated(scenario.state, scenario.config)
        with self._edit_lock():
            self._setup(scenario)
            with self.cars._lock:
                self.cars._owners.clear()
                self.cars._last.clear()
                self.cars._statuses.clear()
            self._generation += 1
            self._world = replace(self._world, captured_at=time.time(), valid=True)
            self.gps._update(self._world)
        self.events.publish('lifecycle', dict(kind='scenario_loaded', generation=self._generation))
        return self.gps.state()

    @rpc_method
    def scenario(self):
        with self._world_lock:
            motion = []
            for kind, bodies in [('car', self._physics.cars), ('ball', self._physics.balls)]:
                for number, body in bodies.items():
                    motion.append(BodyMotion(kind, number, body.velocity.x, body.velocity.y, math.degrees(body.angular_velocity)))
            return SimScenario(deepcopy(self._world), motion, deepcopy(self.config), self.seed, self._waiting)

    @rpc_method
    def update_state(self, patch):
        if isinstance(patch, str): patch = decode(json.loads(patch))
        if not isinstance(patch, StatePatch): raise TypeError('expected StatePatch')
        with self._edit_lock():
            if (any(not isinstance(c, CarState) for c in patch.cars) or any(not isinstance(b, BallState) for b in patch.balls)):
                raise ValueError('invalid patch objects')
            cars, balls = {c.id: c for c in patch.cars}, {b.id: b for b in patch.balls}
            if len(cars) != len(patch.cars) or len(balls) != len(patch.balls):
                raise ValueError('invalid or duplicate patch objects')
            if not set(cars) <= set(self._physics.cars) or not set(balls) <= set(self._physics.balls):
                raise ValueError('unknown patch object')
            owner = session_context.get()
            for key in [('car', n) for n in cars] + [('ball', n) for n in balls]:
                if key in self._held and self._held[key][0] != owner: raise RuntimeError('object held by another editor')
            candidate = self._validated(replace(self._world,
                cars=[cars.get(c.id, c) for c in self._world.cars], balls=[balls.get(b.id, b) for b in self._world.balls],
                captured_at=time.time(), sequence=self._world.sequence+1), self.config)
            for number in cars:
                self._cancel(number)
                self._velocities.pop(number, None)
                with self.cars._lock:
                    self.cars._owners.pop(number, None)
                    self.cars._last.pop(number, None)
            for car in cars.values(): self._physics.edit('car', car)
            for ball in balls.values(): self._physics.edit('ball', ball)
            self._world = candidate
            self.gps._update(candidate)
        self.events.publish('lifecycle', dict(kind='state_edited', patch=patch))
        return candidate

    @rpc_method
    def edit(self, operation, kind, number, pose=None, *, source_id, token=None):
        if source_id != self.source_id: raise ValueError('editor belongs to another provider')
        if kind not in ('car', 'ball') or operation not in ('grab', 'move', 'release'): raise ValueError('invalid gesture')
        owner, key = session_context.get(), (kind, number)
        with self._edit_lock():
            bodies = self._physics.cars if kind == 'car' else self._physics.balls
            if number not in bodies: raise ValueError('unknown edit object')
            held = self._held.get(key)
            if operation == 'grab':
                if held is not None: raise RuntimeError('object already grabbed')
                token = uuid.uuid4().hex
                self._held[key] = (owner, token)
                if kind == 'car':
                    self._cancel(number)
                    with self.cars._lock:
                        self.cars._owners.pop(number, None)
                        self.cars._last.pop(number, None)
                self._physics.hold(kind, number, True)
            else:
                if held != (owner, token): raise ValueError('stale or foreign grab')
                if operation == 'release':
                    self._physics.hold(kind, number, False)
                    self._held.pop(key)
                else:
                    if not isinstance(pose, Pose): raise TypeError('expected Pose')
                    if kind == 'car':
                        c, s = math.cos(math.radians(pose.angle)), math.sin(math.radians(pose.angle))
                        xs = [c*x-s*y for x, y in self.config['body']]
                        ys = [s*x+c*y for x, y in self.config['body']]
                    else:
                        r = self.config['ball_radius']
                        xs = ys = [-r, r]
                    x = max(-self._world.field.width/2-min(xs), min(self._world.field.width/2-max(xs), pose.x))
                    y = max(-self._world.field.height/2-min(ys), min(self._world.field.height/2-max(ys), pose.y))
                    obj = CarState(number, Pose(x, y, pose.angle)) if kind == 'car' else BallState(number, Pose(x, y))
                    self.update_state(StatePatch(cars=[obj]) if kind == 'car' else StatePatch(balls=[obj]))
            self.events.publish('lifecycle', dict(kind='simulation_edit', operation=operation, object_kind=kind, id=number))
            return dict(token=token, generation=self._generation)

    def disconnected(self, session):
        with self._edit_lock():
            for (kind, number), (owner, token) in list(self._held.items()):
                if owner == session:
                    self._physics.hold(kind, number, False)
                    self._held.pop((kind, number))

    @rpc_method
    def set_feedback(self, enabled):
        if type(enabled) is not bool: raise ValueError('enabled must be boolean')
        with self._world_lock:
            self._feedback_enabled = enabled
            if not enabled: self._feedback_pending.clear(); self._deliveries.clear()

    def step(self, dt):
        if not math.isfinite(dt) or not 0 < dt <= .1: raise ValueError('step must be 0..0.1 seconds')
        with self._world_lock:
            self._time += dt
            for number, (due, operation, value) in list(self._pending.items()):
                if due <= self._time + 1e-12: self._activate(number, operation, value)
            for due, captured, state in list(self._feedback_pending):
                if due <= self._time + 1e-12:
                    self._feedback, self._feedback_at = state, captured
                    self._feedback_pending.pop(0)
            self._navigate()
            if not self._waiting:
                commands = {}
                for number, velocity in self._velocities.items():
                    setting = self._configuration[number]
                    trim = AxisTrim(setting['trim_matrix'], mixed=setting['mixed_trim'])
                    commands[number] = Velocity(*trim.apply((velocity.x, velocity.y, velocity.w)))
                self._physics.advance(dt, commands)
            self._world = replace(self._physics.snapshot(self._world), sequence=self._world.sequence+1,
                                  captured_at=self._source_origin+self._time, received_at=time.time(), valid=True)
            if self._feedback_enabled:
                delay = self.config['latency']['udp_ms']/1000
                self._feedback_pending.append((self._time+delay, self._time, self._world))
                self._deliveries.append((self._time+delay, self._world))
            while self._deliveries and self._deliveries[0][0] <= self._time + 1e-12:
                _, delivered = self._deliveries.pop(0)
                self.gps._update(replace(delivered, source_age=self.config['latency']['udp_ms']/1000,
                                         received_at=time.time()))

    def start(self):
        if self._thread: return self
        def run():
            last = time.monotonic()
            while not self._stop.wait(1/self._rate):
                now = time.monotonic()
                self.step(min(now-last, .1))
                last = now
        self._thread = Thread(target=run, daemon=True)
        self._thread.start()
        return self

    def close(self):
        self._stop.set()
        if self._thread: self._thread.join(timeout=2)
        super().close()


def main(argv=None):
    from .cli import provider_parser, run_provider
    parser = provider_parser(__doc__)
    parser.add_argument('scenario', nargs='?')
    parser.add_argument('--scenario', dest='named_scenario')
    parser.add_argument('--load', metavar='FILE')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--ball-model', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--wait-for-control', action='store_true', default=None)
    parser.add_argument('--save-scenario', type=Path)
    parser.add_argument('--rate', type=float, default=30)
    args = parser.parse_args(argv)
    name = args.named_scenario or args.scenario
    if args.scenario and args.named_scenario: parser.error('choose one scenario argument')
    if args.load:
        if name or args.config or args.ball_model or args.save_scenario: parser.error('--load only takes a scenario file')
        with VSSSProvider.connect(args.endpoint) as client:
            client.load_scenario(read_scenario(args.load))
        print(f'Loaded {args.load} into the running simulator; commands cleared', flush=True)
        return
    scenario = goalie_scenario(args.seed) if name == 'goalie' else read_scenario(name) if name else None
    config = json.loads(args.config.read_text()) if args.config else {}
    if args.ball_model:
        model = json.loads(args.ball_model.read_text())
        config['ball_model'] = model.get('ball_model', model.get('model', model))
    sim = VSSSSim(scenario, config=config, seed=args.seed, rate=args.rate, wait_for_control=args.wait_for_control)
    if args.save_scenario:
        save_scenario(sim.scenario(), args.save_scenario)
        print('Saved', args.save_scenario)
    run_provider(sim.start(), args)


if __name__ == '__main__': main()
