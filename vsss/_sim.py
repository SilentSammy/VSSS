"""Seedable planar simulation, independent of networking and wall-clock time.

Cars are driven convex bodies; balls use the measured impact/rolling model.
Car contacts are kinematic constraints, not a calibrated mass/inertia solver.
"""
from dataclasses import dataclass, field
from collections import deque
from copy import deepcopy
from importlib.resources import files
import json
import math
import random

from mp.onboard_controller import DEFAULT_CONTROLLER, OnboardController


def finite(value, name, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f'{name} is outside {minimum}..{maximum}')
    return float(value)


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def subtract(a, b):
    return a[0] - b[0], a[1] - b[1]


def polygon(vertices):
    """Strictly convex CCW body containing the pose origin."""
    if not isinstance(vertices, list) or not 3 <= len(vertices) <= 32:
        raise ValueError('body must contain 3..32 convex CCW vertices')
    result = []
    for point in vertices:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError('body vertices must be XY pairs')
        result.append(tuple(finite(v, 'body coordinate', -1, 1) for v in point))
    for i, a in enumerate(result):
        b = result[(i + 1) % len(result)]
        edge = subtract(b, a)
        if cross(edge, (-a[0], -a[1])) <= 1e-9 or any(
                cross(edge, subtract(p, a)) <= 1e-9 for j, p in enumerate(result)
                if j not in (i, (i + 1) % len(result))):
            raise ValueError('body must be strictly convex, CCW, and contain the origin')
    return tuple(result)


def seed_from_bag(config, path, motion=False):
    """Read header/first frame, or the first two frames when motion is requested."""
    snapshot = {}
    first = None
    with open(path, encoding='utf-8') as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError as error:
                raise ValueError(f'invalid JSON on bag line {number}') from error
            if not isinstance(event, dict):
                raise ValueError(f'bag line {number} must be an object')
            if event.get('type') == 'bag_header':
                snapshot = event.get('provider') or {}
            elif event.get('type') == 'provider_snapshot':
                snapshot = event
            if event.get('type') != 'udp' or not event.get('valid'):
                continue
            packet = event.get('packet') or {}
            if not packet.get('board_detected'):
                continue
            if first is not None:
                from ._sim_scenarios import apply_frames
                apply_frames(config, first, dict(time=packet.get('timestamp', event.get('wall_time_ns', 0) / 1e9),
                                                cars=packet.get('markers', []), balls=packet.get('balls', [])))
                return
            setup = (((snapshot.get('vision') or {}).get('metadata') or {}).get('status') or {}).get('setup')
            if setup:
                recorded = (setup.get('simulation') or {}).get('config') or {}
                config.update({key: deepcopy(value) for key, value in recorded.items() if key in config})
                visual = deepcopy(setup)
                visual.pop('simulation', None)  # Do not nest old snapshots on repeated round-trips.
                config['vision_setup'] = visual
                config['field'] = list(setup['board']['dimensions_m'])
                if setup.get('ball_detector'):
                    config['ball_radius'] = setup['ball_detector']['radius_m']
            config['cars'] = [dict(id=car['id'], x=car['x'], y=car['y'], angle=car['angle'])
                              for car in packet.get('markers', [])]
            config['balls'] = [dict(x=ball['x'], y=ball['y']) for ball in packet.get('balls', [])]
            if not config['cars']:
                raise ValueError('first usable bag frame has no cars')
            if not motion:
                return
            first = dict(time=packet.get('timestamp', event.get('wall_time_ns', 0) / 1e9),
                         cars=packet.get('markers', []), balls=packet.get('balls', []))
    if first is not None:
        raise ValueError('motion initialization requires two usable field frames')
    raise ValueError('bag contains no valid frame with a detected board')


def load_config(path=None, ball_model=None, bag=None, scenario=None, seed=None):
    config = json.loads(files('vsss').joinpath('sim_defaults.json').read_text(encoding='utf-8'))
    if bag is not None and scenario is not None:
        raise ValueError('choose a bag or a scenario, not both')
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2 ** 63):
        raise ValueError('seed must be an integer in 0..2**63-1')
    if scenario is not None:
        from ._sim_scenarios import seed_from_scenario
        seed_from_scenario(config, scenario, seed)
    if bag is not None:
        try:
            seed_from_bag(config, bag)
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError(f'invalid bag setup or first frame: {error}') from error
    if path is not None:
        with open(path, encoding='utf-8') as source:
            overrides = json.load(source)
        if not isinstance(overrides, dict) or overrides.keys() - config.keys():
            raise ValueError('simulation config must be an object with known settings')
        for key, value in overrides.items():
            if key in ('motion', 'ball_model', 'ball_wobble', 'latency') and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
    if ball_model is not None:
        with open(ball_model, encoding='utf-8') as source:
            model = json.load(source)
        if not isinstance(model, dict):
            raise ValueError('ball model must be a JSON object or fitted report')
        config['ball_model'] = model.get('model', model)
    if seed is not None:
        config['random_seed'] = seed
    return config


@dataclass(frozen=True)
class BallModel:
    impact_gain: float
    rolling_deceleration: float
    drift_acceleration_x: float = 0.
    drift_acceleration_y: float = 0.
    stop_speed: float = .05

    def __post_init__(self):
        finite(self.impact_gain, 'impact_gain', 0, 2)
        finite(self.rolling_deceleration, 'rolling_deceleration', 1e-6, 100)
        finite(self.stop_speed, 'stop_speed', 0, 10)
        finite(self.drift_acceleration_x, 'drift_acceleration_x')
        finite(self.drift_acceleration_y, 'drift_acceleration_y')
        if math.hypot(self.drift_acceleration_x, self.drift_acceleration_y) >= self.rolling_deceleration:
            raise ValueError('drift acceleration must be smaller than rolling resistance')

    def impact_velocity(self, car_velocity, ball_velocity, normal):
        impulse = self.impact_gain * max(0., dot(subtract(car_velocity, ball_velocity), normal))
        return tuple(ball_velocity[i] + impulse * normal[i] for i in range(2))

    def roll(self, position, velocity, dt, *, stop_speed=None):
        speed = math.hypot(*velocity)
        if speed <= (self.stop_speed if stop_speed is None else stop_speed):
            return position, (0., 0.)
        acceleration = (-self.rolling_deceleration * velocity[0] / speed + self.drift_acceleration_x,
                        -self.rolling_deceleration * velocity[1] / speed + self.drift_acceleration_y)
        next_velocity = tuple(velocity[i] + acceleration[i] * dt for i in range(2))
        if dot(velocity, next_velocity) <= 0:
            # Integrate only until rest; static friction then holds the ball.
            stop_dt = min(dt, speed / max(1e-9, -dot(acceleration, velocity) / speed))
            return tuple(position[i] + velocity[i] * stop_dt / 2 for i in range(2)), (0., 0.)
        return tuple(position[i] + (velocity[i] + next_velocity[i]) * dt / 2
                     for i in range(2)), next_velocity

    def stopping_point(self, position, velocity):
        # Keep the analysis tool's original endpoint integration for reproducibility.
        for _ in range(1500):
            if math.hypot(*velocity) <= self.stop_speed:
                break
            next_position, next_velocity = self.roll(position, velocity, .01)
            if dot(velocity, next_velocity) <= 0:
                break
            position, velocity = next_position, next_velocity
        return tuple(position)


@dataclass(frozen=True)
class BallWobble:
    """Conservative low-speed roughness, separate from the historical fitted model."""
    enabled: bool = True
    onset_speed: float = .14
    acceleration: float = .06
    correlation_time: float = .25
    drag_variation: float = .35
    slow_drag_scale: float = .4
    stop_speed: float = .01

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('ball_wobble.enabled must be a boolean')
        finite(self.onset_speed, 'wobble onset_speed', .002, 2)
        finite(self.stop_speed, 'wobble stop_speed', .001, self.onset_speed - .0001)
        finite(self.acceleration, 'wobble acceleration', 0, 2)
        finite(self.correlation_time, 'wobble correlation_time', .02, 5)
        finite(self.drag_variation, 'wobble drag_variation', 0, .8)
        finite(self.slow_drag_scale, 'wobble slow_drag_scale', .1, 1)


@dataclass
class SimCar:
    number: int
    x: float
    y: float
    angle: float
    command: tuple = (0., 0., 0.)
    vx: float = 0.
    vy: float = 0.
    omega: float = 0.
    mode: int = 0
    fault: int = 0
    trim: tuple = (1., 0., 0., 0., 1., 0., 0., 0., 1.)
    controller: dict = field(default_factory=lambda: dict(DEFAULT_CONTROLLER))
    navigator: object = field(default_factory=OnboardController)
    waypoint_started: float = 0.

    def stop(self):
        self.command = (0., 0., 0.)
        self.vx = self.vy = self.omega = 0.
        self.mode = self.fault = 0
        self.navigator.clear_target()


@dataclass
class SimBall:
    x: float
    y: float
    vx: float = 0.
    vy: float = 0.
    wobble_lateral: float = 0.
    wobble_drag: float = 0.
    rng: object = field(default=None, repr=False, compare=False)

    def settle(self):
        self.vx = self.vy = self.wobble_lateral = self.wobble_drag = 0.


def ball_contact(vertices, center, radius):
    """Return outward normal, penetration, and surface point, or None."""
    inside = True
    closest = None
    best = float('inf')
    for i, a in enumerate(vertices):
        b = vertices[(i + 1) % len(vertices)]
        edge = subtract(b, a)
        length2 = dot(edge, edge)
        inside &= cross(edge, subtract(center, a)) >= 0
        t = max(0., min(1., dot(subtract(center, a), edge) / length2))
        point = (a[0] + t * edge[0], a[1] + t * edge[1])
        delta = subtract(center, point)
        distance = math.hypot(*delta)
        if distance < best:
            best, closest = distance, point
            normal = (edge[1] / math.sqrt(length2), -edge[0] / math.sqrt(length2))
    if not inside and best >= radius:
        return None
    if best > 1e-10:
        direction = subtract(closest, center) if inside else subtract(center, closest)
        normal = tuple(v / best for v in direction)
    return normal, radius + best if inside else radius - best, closest


def polygon_contact(a, b):
    """Separating-axis contact: normal points from A to B."""
    depth = float('inf')
    normal = None
    for vertices in (a, b):
        for i, p in enumerate(vertices):
            edge = subtract(vertices[(i + 1) % len(vertices)], p)
            axis = (-edge[1], edge[0])
            norm = math.hypot(*axis)
            axis = tuple(v / norm for v in axis)
            pa, pb = [dot(v, axis) for v in a], [dot(v, axis) for v in b]
            if max(pa) <= min(pb) or max(pb) <= min(pa):
                return None
            forward, backward = max(pa) - min(pb), max(pb) - min(pa)
            overlap = min(forward, backward)
            if overlap < depth:
                depth = overlap
                normal = axis if forward < backward else tuple(-v for v in axis)
    return normal, depth


class SimWorld:
    def __init__(self, config):
        self.config = deepcopy(config)
        try:
            self._configure(self.config)
        except (TypeError, KeyError, IndexError, AttributeError) as error:
            raise ValueError(f'invalid simulation configuration: {error}') from error

    def _configure(self, config):
        if len(config['field']) != 2:
            raise ValueError('field must be [width, height]')
        self.width, self.height = (finite(v, 'field dimension', .3, 100) for v in config['field'])
        self.body = polygon(config['body'])
        self.radius = finite(config['ball_radius'], 'ball_radius', .001, .2)
        if 2 * self.radius >= min(self.width, self.height):
            raise ValueError('ball does not fit in the field')
        self.model = BallModel(**config['ball_model'])
        self.wobble = BallWobble(**config.get('ball_wobble', {}))
        seed = config.get('random_seed')
        if seed is None:
            seed = random.SystemRandom().randrange(2 ** 63)
        if type(seed) is not int or not 0 <= seed < 2 ** 63:
            raise ValueError('random_seed must be null or an integer in 0..2**63-1')
        self.seed = config['random_seed'] = seed
        self.waiting_for_control = config.get('wait_for_control', False)
        if type(self.waiting_for_control) is not bool:
            raise ValueError('wait_for_control must be a boolean')
        motion = config['motion']
        if set(motion) != {'max_speed', 'max_yaw_rate', 'response_time'}:
            raise ValueError('unknown or missing motion settings')
        self.max_speed = finite(motion['max_speed'], 'max_speed', .01, 5)
        self.max_yaw = math.radians(finite(motion['max_yaw_rate'], 'max_yaw_rate', 1, 720))
        self.response = finite(motion['response_time'], 'response_time', .001, 10)
        self.physics_hz = finite(config['physics_hz'], 'physics_hz', 50, 1000)
        self.telemetry_hz = finite(config['telemetry_hz'], 'telemetry_hz', 1, self.physics_hz)
        self.restitution = finite(config['wall_restitution'], 'wall_restitution', 0, 1)
        latency = config.get('latency', dict(udp_ms=100, ble_ms=40))
        if set(latency) != {'udp_ms', 'ble_ms'}:
            raise ValueError('latency requires udp_ms and ble_ms')
        self.udp_latency = finite(latency['udp_ms'], 'udp_ms', 0, 400) / 1000
        self.ble_latency = finite(latency['ble_ms'], 'ble_ms', 0, 1000) / 1000
        # Bound surface travel per fixed step to limit contact tunnelling.
        body_radius = max(math.hypot(*v) for v in self.body)
        if 2 * body_radius >= min(self.width, self.height):
            raise ValueError('body must fit in the field at every heading')
        self.substeps = max(1, math.ceil((self.max_speed * 2 + self.max_yaw * body_radius) /
                                         self.physics_hz / (self.radius / 2)))
        self.cars = {}
        if not isinstance(config['cars'], list) or not 1 <= len(config['cars']) <= 32:
            raise ValueError('cars must contain 1..32 car definitions')
        for row in config['cars']:
            if not {'id', 'x', 'y', 'angle'} <= set(row) or set(row) - {
                    'id', 'x', 'y', 'angle', 'vx', 'vy', 'angular_velocity_deg_s'}:
                raise ValueError('each car requires id, x, y, angle and optional initial velocities')
            number = row['id']
            if type(number) is not int or not 0 <= number <= 65535 or number in self.cars:
                raise ValueError('car IDs must be unique integers in 0..65535')
            car = SimCar(number, finite(row['x'], 'car x'), finite(row['y'], 'car y'),
                         (finite(row['angle'], 'car angle') + 180) % 360 - 180)
            car.vx = finite(row.get('vx', 0), 'car vx', -5, 5)
            car.vy = finite(row.get('vy', 0), 'car vy', -5, 5)
            car.omega = math.radians(finite(row.get('angular_velocity_deg_s', 0), 'car angular velocity', -720, 720))
            vertices = self.vertices(car)
            if any(abs(x) > self.width / 2 or abs(y) > self.height / 2 for x, y in vertices):
                raise ValueError('initial car body is outside the field')
            if any(polygon_contact(vertices, self.vertices(other)) for other in self.cars.values()):
                raise ValueError('initial car bodies overlap')
            self.cars[number] = car
        # Two-frame starts can contain residual motion faster than a new command.
        travel = max(self.max_speed * 2 + self.max_yaw * body_radius,
                     max(2 * math.hypot(c.vx, c.vy) + abs(c.omega) * body_radius for c in self.cars.values()))
        self.substeps = max(self.substeps, math.ceil(travel / self.physics_hz / (self.radius / 2)))
        self.balls = []
        if not isinstance(config['balls'], list) or len(config['balls']) > 32:
            raise ValueError('balls must be a list of at most 32 balls')
        for row in config['balls']:
            if set(row) - {'x', 'y', 'vx', 'vy'}:
                raise ValueError('unknown ball setting')
            ball = SimBall(finite(row['x'], 'ball x'), finite(row['y'], 'ball y'),
                           finite(row.get('vx', 0), 'ball vx', -5, 5),
                           finite(row.get('vy', 0), 'ball vy', -5, 5))
            if abs(ball.x) + self.radius > self.width / 2 or abs(ball.y) + self.radius > self.height / 2:
                raise ValueError('initial ball is outside the field')
            if any(ball_contact(self.vertices(c), (ball.x, ball.y), self.radius)
                   for c in self.cars.values()) or any(math.hypot(ball.x - b.x, ball.y - b.y) < 2 * self.radius
                                                       for b in self.balls):
                raise ValueError('initial balls overlap another body')
            self.balls.append(ball)
            # Separate streams keep one ball's contacts from changing another's noise.
            ball.rng = random.Random(seed + (len(self.balls) - 1) * 0x9e3779b97f4a7c15)
        self.time = 0.
        self.sequence = 0
        self.next_control = 0.
        self.pending_feedback = deque()
        self.feedback = {}
        self.feedback_sequence = self.capture_sequence = 0
        self.held = {}
        if config.get('vision_setup') is not None and not isinstance(config['vision_setup'], dict):
            raise ValueError('vision_setup must be a metadata object or null')

    def vertices(self, car):
        theta = math.radians(car.angle)
        c, s = math.cos(theta), math.sin(theta)
        return tuple((car.x + c * x - s * y, car.y + s * x + c * y) for x, y in self.body)

    def waypoint(self, car, target):
        if len(target) != 3:
            raise ValueError('waypoint requires x, y, angle')
        x, y, angle = (finite(v, 'waypoint') for v in target)
        if abs(x) > min(.8, self.width / 2) or abs(y) > min(1.5, self.height / 2):
            raise ValueError('waypoint is outside the configured field')
        car.navigator.set_target(x, y, math.radians(angle))
        car.mode, car.fault, car.waypoint_started = 2, 0, self.time

    def status(self, car):
        pose = self.feedback.get(car.number)
        return [car.mode, 3 if pose else 1, car.fault, 0,
                *(pose if pose else (0., 0., 0.)), self.feedback_sequence]

    def step(self, dt):
        finite(dt, 'time step', 1e-6, .1)
        count = max(1, math.ceil(dt * self.physics_hz - 1e-9)) * self.substeps
        for _ in range(count):
            self._step(dt / count)

    def _step(self, dt):
        self.time += dt
        for car in self.cars.values():
            if ('car', car.number) in self.held:
                car.stop()
        control_tick = self.time >= self.next_control
        if control_tick:
            self.next_control += 1 / self.telemetry_hz
            self.capture_sequence += 1
            self.pending_feedback.append((self.time + self.udp_latency, self.capture_sequence,
                {c.number: (c.x, c.y, c.angle) for c in self.cars.values()}))
        while self.pending_feedback and self.pending_feedback[0][0] <= self.time + 1e-9:
            _, self.feedback_sequence, self.feedback = self.pending_feedback.popleft()
        if self.waiting_for_control:
            return  # Keep publishing fresh stationary feedback until a student starts control.
        if control_tick:
            for car in self.cars.values():
                if car.mode != 2:
                    continue
                now = round(self.time * 1000)
                car.command = car.navigator.update(self.feedback.get(car.number), now)
                if car.navigator.reached(now):
                    car.stop()
                    car.mode = 3
                elif (self.time - car.waypoint_started) * 1000 >= car.controller['waypoint_timeout_ms']:
                    car.stop()
                    car.mode, car.fault = 4, 2
        for car in self.cars.values():
            command = [max(-1., min(1., sum(car.trim[3 * i + j] * car.command[j]
                                           for j in range(3)))) for i in range(3)]
            # This is the same saturation applied to the four mecanum wheels.
            scale = max(1., abs(command[0]) + abs(command[1]) + abs(command[2]))
            x, y, w = (v / scale for v in command)
            theta = math.radians(car.angle)
            c, s = math.cos(theta), math.sin(theta)
            target = ((c * x - s * y) * self.max_speed,
                      (s * x + c * y) * self.max_speed, w * self.max_yaw)
            alpha = -math.expm1(-dt / self.response)
            car.vx += alpha * (target[0] - car.vx)
            car.vy += alpha * (target[1] - car.vy)
            car.omega += alpha * (target[2] - car.omega)
            car.x += car.vx * dt
            car.y += car.vy * dt
            car.angle = (car.angle + math.degrees(car.omega * dt) + 180) % 360 - 180
        # Iterative positional constraints handle corners and small stacks.
        cars = list(self.cars.values())
        for _ in range(4):
            for i, car in enumerate(cars):
                self._car_walls(car)
                for other in cars[i + 1:]:
                    contact = polygon_contact(self.vertices(car), self.vertices(other))
                    if contact:
                        normal, depth = contact
                        a = 0 if ('car', car.number) in self.held else 1
                        b = 0 if ('car', other.number) in self.held else 1
                        if a + b == 0:
                            continue
                        car.x -= normal[0] * depth * a / (a + b)
                        car.y -= normal[1] * depth * a / (a + b)
                        other.x += normal[0] * depth * b / (a + b)
                        other.y += normal[1] * depth * b / (a + b)
                        closing = dot((car.vx - other.vx, car.vy - other.vy), normal)
                        if closing > 0:
                            car.vx -= normal[0] * closing * a / (a + b)
                            car.vy -= normal[1] * closing * a / (a + b)
                            other.vx += normal[0] * closing * b / (a + b)
                            other.vy += normal[1] * closing * b / (a + b)
            for car in cars:
                self._car_walls(car)
        # Fast balls get additional integration/contact substeps too.
        ball_steps = max(1, math.ceil(max((math.hypot(b.vx, b.vy) for b in self.balls), default=0)
                                      * dt / (self.radius / 2)))
        for _ in range(ball_steps):
            for index, ball in enumerate(self.balls):
                if ('ball', index) in self.held:
                    ball.settle()
                    continue
                self._roll_ball(ball, dt / ball_steps)
            for _ in range(3):
                for index, ball in enumerate(self.balls):
                    for car in cars:
                        contact = ball_contact(self.vertices(car), (ball.x, ball.y), self.radius)
                        if contact:
                            normal, depth, point = contact
                            if ('ball', index) in self.held:
                                if ('car', car.number) not in self.held:
                                    car.x -= normal[0] * depth
                                    car.y -= normal[1] * depth
                                    self._car_walls(car)
                                continue
                            surface_velocity = (car.vx - car.omega * (point[1] - car.y),
                                                car.vy + car.omega * (point[0] - car.x))
                            ball.vx, ball.vy = self.model.impact_velocity(
                                surface_velocity, (ball.vx, ball.vy), normal)
                            ball.x += normal[0] * depth
                            ball.y += normal[1] * depth
                    self._ball_walls(ball)
                self._ball_pairs()

    def _roll_ball(self, ball, dt):
        speed = math.hypot(ball.vx, ball.vy)
        wobble = self.wobble
        position, velocity = (ball.x, ball.y), (ball.vx, ball.vy)
        if not wobble.enabled or speed >= wobble.onset_speed:
            ball.wobble_lateral = ball.wobble_drag = 0.
            (ball.x, ball.y), (ball.vx, ball.vy) = self.model.roll(
                position, velocity, dt, stop_speed=wobble.stop_speed if wobble.enabled else None)
            return
        if speed <= wobble.stop_speed:
            ball.settle()
            return
        next_position, next_velocity = self.model.roll(position, velocity, dt, stop_speed=wobble.stop_speed)
        base_speed = min(speed, math.hypot(*next_velocity))
        if base_speed == 0:
            ball.x, ball.y = next_position
            ball.settle()
            return
        # Exact OU update: smooth, correlated acceleration rather than position jitter.
        retention = math.exp(-dt / wobble.correlation_time)
        innovation = math.sqrt(-math.expm1(-2 * dt / wobble.correlation_time))
        ball.wobble_lateral = max(-2., min(2., retention * ball.wobble_lateral + innovation * ball.rng.gauss(0, 1)))
        ball.wobble_drag = max(-1., min(1., retention * ball.wobble_drag + innovation * ball.rng.gauss(0, 1)))
        weight = min(1., (wobble.onset_speed - speed) / (wobble.onset_speed - wobble.stop_speed))
        braking_scale = (1 - weight * (1 - wobble.slow_drag_scale)) * (
            1 + weight * wobble.drag_variation * ball.wobble_drag)
        next_speed = max(0., speed - (speed - base_speed) * braking_scale)
        fade = min(1., (speed - wobble.stop_speed) / wobble.stop_speed)
        turn = wobble.acceleration * weight * fade * ball.wobble_lateral * dt / speed
        angle = math.atan2(next_velocity[1], next_velocity[0]) + turn
        if next_speed <= wobble.stop_speed:
            next_speed = 0.
        ball.vx, ball.vy = next_speed * math.cos(angle), next_speed * math.sin(angle)
        ball.x += (velocity[0] + ball.vx) * dt / 2
        ball.y += (velocity[1] + ball.vy) * dt / 2
        if next_speed == 0:
            ball.settle()

    def _car_walls(self, car):
        vertices = self.vertices(car)
        dx = max(0., -self.width / 2 - min(x for x, y in vertices)) + min(
            0., self.width / 2 - max(x for x, y in vertices))
        dy = max(0., -self.height / 2 - min(y for x, y in vertices)) + min(
            0., self.height / 2 - max(y for x, y in vertices))
        car.x += dx
        car.y += dy
        if dx and car.vx * dx < 0:
            car.vx = 0.
        if dy and car.vy * dy < 0:
            car.vy = 0.

    def _ball_walls(self, ball):
        for position, velocity, half in (('x', 'vx', self.width / 2), ('y', 'vy', self.height / 2)):
            limit = half - self.radius
            value, speed = getattr(ball, position), getattr(ball, velocity)
            if abs(value) > limit:
                sign = 1 if value > 0 else -1
                setattr(ball, position, sign * limit)
                if speed * sign > 0:
                    setattr(ball, velocity, -speed * self.restitution)

    def _ball_pairs(self):
        for i, a in enumerate(self.balls):
            for j, b in enumerate(self.balls[i + 1:], i + 1):
                delta = b.x - a.x, b.y - a.y
                distance = math.hypot(*delta)
                if distance >= 2 * self.radius:
                    continue
                normal = tuple(v / distance for v in delta) if distance > 1e-10 else (1., 0.)
                wa = 0 if ('ball', i) in self.held else 1
                wb = 0 if ('ball', j) in self.held else 1
                if wa + wb == 0:
                    continue
                depth = (2 * self.radius - distance) / (wa + wb)
                a.x -= normal[0] * depth * wa
                a.y -= normal[1] * depth * wa
                b.x += normal[0] * depth * wb
                b.y += normal[1] * depth * wb
                impulse = (1 + self.restitution) * max(0., dot((a.vx - b.vx, a.vy - b.vy), normal)) / (wa + wb)
                a.vx -= impulse * normal[0] * wa
                a.vy -= impulse * normal[1] * wa
                b.vx += impulse * normal[0] * wb
                b.vy += impulse * normal[1] * wb

    def object_state(self, kind, number):
        obj = self.cars[number] if kind == 'car' else self.balls[number]
        state = dict(kind=kind, id=number, x=obj.x, y=obj.y, held=(kind, number) in self.held)
        if kind == 'car':
            state['angle'] = obj.angle
        return state

    def edit(self, owner, message):
        """Apply an atomic editor request; caller holds the world lock."""
        if not isinstance(message, dict):
            raise ValueError('edit request must be an object')
        if message == {'operation': 'state'}:
            return dict(cars=[self.object_state('car', n) for n in self.cars],
                        balls=[self.object_state('ball', n) for n in range(len(self.balls))])
        operation, kind, number = (message.get(k) for k in ('operation', 'kind', 'id'))
        if operation not in ('move', 'grab', 'release') or kind not in ('car', 'ball'):
            raise ValueError('expected move/grab/release of a car or ball')
        allowed = {'operation', 'kind', 'id'} | ({'x', 'y', 'angle'} if operation == 'move' else set())
        if message.keys() - allowed or (kind == 'ball' and 'angle' in message):
            raise ValueError('unknown edit settings')
        if type(number) is not int or (kind == 'car' and number not in self.cars) or (
                kind == 'ball' and not 0 <= number < len(self.balls)):
            raise ValueError('unknown object ID (balls use zero-based indices)')
        key = kind, number
        if key in self.held and self.held[key] != owner:
            raise ValueError('object is being dragged by another editor')
        if operation == 'release' and key not in self.held:
            raise ValueError('object is not held by this editor')
        obj = self.cars[number] if kind == 'car' else self.balls[number]
        if operation == 'move':
            x, y = finite(message.get('x'), 'x'), finite(message.get('y'), 'y')
            if kind == 'car':
                angle = (finite(message.get('angle', obj.angle), 'angle') + 180) % 360 - 180
                theta = math.radians(angle)
                c, s = math.cos(theta), math.sin(theta)
                offsets = [(c * a - s * b, s * a + c * b) for a, b in self.body]
                x = max(-self.width / 2 - min(a for a, b in offsets), min(x, self.width / 2 - max(a for a, b in offsets)))
                y = max(-self.height / 2 - min(b for a, b in offsets), min(y, self.height / 2 - max(b for a, b in offsets)))
                obj.angle = angle
            else:
                x = max(-self.width / 2 + self.radius, min(x, self.width / 2 - self.radius))
                y = max(-self.height / 2 + self.radius, min(y, self.height / 2 - self.radius))
            obj.x, obj.y = x, y
        if kind == 'car':
            obj.stop()
        else:
            obj.settle()
        if operation == 'grab':
            self.held[key] = owner
        elif operation == 'release':
            self.held.pop(key, None)
        return self.object_state(kind, number)

    def release_editor(self, owner):
        released = []
        for (kind, number), holder in list(self.held.items()):
            if holder == owner:
                released.append(self.edit(owner, dict(operation='release', kind=kind, id=number)))
        return released

    def packet(self, timestamp=None):
        self.sequence += 1
        return dict(board_detected=True, timestamp=self.time if timestamp is None else timestamp,
                    frame_sequence=self.sequence, frame_age_ms=0.,
                    markers=[dict(id=c.number, x=c.x, y=c.y, angle=c.angle) for c in self.cars.values()],
                    balls=[dict(x=b.x, y=b.y) for b in self.balls])

    def setup(self):
        setup = deepcopy(self.config.get('vision_setup') or {})
        setup.setdefault('name', 'VSSS simulation')
        setup.setdefault('marker_size_m', .09)
        setup.setdefault('plot_margin_m', .05)
        board = setup.setdefault('board', {})
        board['dimensions_m'] = [self.width, self.height]
        board.setdefault('print_dimensions_m', [self.width, self.height])
        setup.setdefault('ball_detector', {})
        setup['ball_detector'] = dict(setup['ball_detector'] or {}, radius_m=self.radius)
        setup['simulation'] = dict(body=[list(v) for v in self.body], config=deepcopy(self.config))
        return setup
