"""Portable simulation scenarios and validated physical configuration."""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import random
from .models import BallState, BodyMotion, CarState, Field, Pose, SimScenario, WorldState, decode, encode, finite

DEFAULT_CONFIG = dict(
    body=[[-.06, -.06], [.06, -.06], [.06, .06], [-.06, .06]], ball_radius=.02,
    car_mass=.4, ball_mass=.045, wall_restitution=.6, physics_hz=240,
    motion=dict(max_speed=.6, max_yaw_rate=180., response_time=.06, max_acceleration=3.),
    ball_model=dict(impact_gain=.75, rolling_deceleration=.25, drift_acceleration_x=0.,
                    drift_acceleration_y=0., stop_speed=.005),
    ball_wobble=dict(enabled=False, onset_speed=.14, acceleration=.06, correlation_time=.25,
                     drag_variation=.35, slow_drag_scale=.4, stop_speed=.01),
    latency=dict(udp_ms=0., ble_ms=0.))


def printable_field():
    """The default simulator boundary: the full printable lab board."""
    # Keep vision/OpenCV imports lazy so scenarios that supply a field remain
    # usable without loading camera infrastructure.
    from aruco_gps.vision_setup import DEFAULT_SETUP
    return Field(*DEFAULT_SETUP.board.get_print_dimensions())


def physical_config(value=None):
    if value is not None and not isinstance(value, dict): raise ValueError('config must be a mapping')
    result = deepcopy(DEFAULT_CONFIG)
    for key, item in (value or {}).items():
        if key not in result: raise ValueError(f'unknown simulation setting: {key}')
        if isinstance(result[key], dict):
            if not isinstance(item, dict) or set(item) - set(result[key]): raise ValueError(f'invalid {key} settings')
            result[key].update(item)
        else: result[key] = deepcopy(item)
    for key in ('ball_radius', 'car_mass', 'ball_mass'):
        finite(result[key])
        if not .001 <= result[key] <= 10: raise ValueError(f'invalid {key}')
    finite(result['wall_restitution'], result['physics_hz'])
    if not 0 <= result['wall_restitution'] <= 1 or not 30 <= result['physics_hz'] <= 1000:
        raise ValueError('invalid wall restitution or physics rate')
    body = result['body']
    if not isinstance(body, list) or not 3 <= len(body) <= 32: raise ValueError('body needs 3..32 CCW vertices')
    for point in body:
        if not isinstance(point, (list, tuple)) or len(point) != 2: raise ValueError('invalid body vertex')
        finite(*point)
        if max(abs(v) for v in point) > 1: raise ValueError('body exceeds one metre')
    for i, a in enumerate(body):
        b = body[(i+1) % len(body)]
        cross = lambda p: (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])
        if cross((0, 0)) <= 0 or any(cross(p) <= 0 for j, p in enumerate(body) if j not in (i, (i+1) % len(body))):
            raise ValueError('body must be strictly convex CCW around the origin')
    for group in ('motion', 'ball_model', 'latency'):
        for key, item in result[group].items():
            finite(item)
            if group == 'motion' and not 0 < item <= (1080 if key == 'max_yaw_rate' else 10): raise ValueError(f'invalid {key}')
            if group == 'latency' and not 0 <= item <= 1000: raise ValueError('delay must be 0..1000 ms')
            if group == 'ball_model' and abs(item) > 10: raise ValueError('ball model out of range')
    model = result['ball_model']
    if not 0 <= model['impact_gain'] <= 2 or model['rolling_deceleration'] < 0 or model['stop_speed'] < 0:
        raise ValueError('invalid ball model')
    if math.hypot(model['drift_acceleration_x'], model['drift_acceleration_y']) > model['rolling_deceleration']:
        raise ValueError('drift must not exceed rolling resistance')
    wobble = result['ball_wobble']
    if type(wobble['enabled']) is not bool: raise ValueError('wobble enabled must be boolean')
    for key, item in wobble.items():
        if key == 'enabled': continue
        finite(item)
        if not 0 <= item <= 10: raise ValueError('invalid wobble setting')
    if wobble['correlation_time'] <= 0: raise ValueError('wobble correlation time must be positive')
    return result


def load_scenario(path):
    with Path(path).open(encoding='utf-8') as file:
        first = json.loads(file.readline())
        if first.get('schema') == 'vsss.recording/v1': return SimScenario(decode(first['state']))
        if first.get('type') == 'sim_scenario':
            a, b = json.loads(file.readline()), json.loads(file.readline())
            dt = b['time'] - a['time']
            finite(dt)
            if dt <= 0: raise ValueError('scenario frame times must increase')
            config = dict(first.get('config', {}))
            field = Field(*config.pop('field')) if 'field' in config else printable_field()
            state = WorldState(field=field, cars=[CarState(c['id'], Pose(c['x'], c['y'], c.get('angle', 0))) for c in a['cars']],
                balls=[BallState(c.get('id', i), Pose(c['x'], c['y'])) for i, c in enumerate(a['balls'])])
            motion = []
            for kind, objects, second in [('car', state.cars, b['cars']), ('ball', state.balls, b['balls'])]:
                other = {c.get('id', i): c for i, c in enumerate(second)}
                if len(other) != len(second) or set(other) != {c.id for c in objects}: raise ValueError('scenario frame identities differ')
                for obj in objects:
                    c = other[obj.id]
                    angle = (c.get('angle', 0)-obj.pose.angle+180) % 360-180
                    motion.append(BodyMotion(kind, obj.id, (c['x']-obj.pose.x)/dt, (c['y']-obj.pose.y)/dt, angle/dt))
            return SimScenario(state, motion, config, first.get('seed'), first.get('wait_for_control', False))
        value = decode(first)
        if isinstance(value, WorldState): return SimScenario(value)
        if isinstance(value, SimScenario): return value
        raise ValueError('expected WorldState, SimScenario, two-frame scenario, or recording header')


def save_scenario(scenario, path):
    physical_config(scenario.config)
    with Path(path).open('x', encoding='utf-8') as file:
        file.write(json.dumps(encode(scenario), allow_nan=False) + '\n')


def goalie_scenario(seed=None):
    rng = random.Random(seed)
    x, y, target = rng.uniform(-.55, .55), rng.uniform(-1, 0), rng.uniform(-.55, .55)
    dx, dy = target-x, 1.45-y
    scale = .8 / math.hypot(dx, dy)
    state = WorldState(field=printable_field(), cars=[CarState(1, Pose(0, 1.2))],
                       balls=[BallState(0, Pose(x, y))])
    # The generated shot must reach the goal line; generic illustrative drag
    # would stop a 0.8 m/s ball before some of the longer seeded trajectories.
    config = dict(ball_model=dict(rolling_deceleration=.07489989178563962,
                                 impact_gain=.8803656717213731, stop_speed=.05))
    return SimScenario(state, [BodyMotion('ball', 0, dx*scale, dy*scale)], config, seed=seed, wait_for_control=True)
