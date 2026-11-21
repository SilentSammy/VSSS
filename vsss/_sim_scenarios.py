"""Scenario JSONL: a config header and two timed position frames."""
from copy import deepcopy
import json
import math
from pathlib import Path
import random

from ._sim import finite, seed_from_bag


def apply_frames(config, first, second):
    """Start at the first frame; initial velocity is (second - first) / dt."""
    dt = finite(finite(second['time'], 'second frame time') - finite(first['time'], 'first frame time'),
                'frame interval')
    if dt <= 0:
        raise ValueError('scenario frame times must increase')
    a, b = first['cars'], second['cars']
    if not isinstance(a, list) or not isinstance(b, list):
        raise ValueError('frame cars must be lists')
    if len({c['id'] for c in a}) != len(a) or len({c['id'] for c in b}) != len(b):
        raise ValueError('frame car IDs must be unique')
    by_id = {c['id']: c for c in b}
    if set(by_id) != {c['id'] for c in a}:
        raise ValueError('both frames must contain the same car IDs')
    cars = []
    for car in a:
        other = by_id[car['id']]
        result = dict(id=car['id'], **{axis: finite(car[axis], 'car ' + axis) for axis in ('x', 'y', 'angle')})
        for axis in ('x', 'y'):
            result['v' + axis] = (finite(other[axis], 'car ' + axis) - result[axis]) / dt
        turn = (finite(other['angle'], 'car angle') - result['angle'] + 180) % 360 - 180
        result['angular_velocity_deg_s'] = turn / dt
        cars.append(result)
    a, b = first['balls'], second['balls']
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b):
        raise ValueError('both frames must contain the same number of balls, in the same order')
    balls = []
    for ball, other in zip(a, b):
        result = {axis: finite(ball[axis], 'ball ' + axis) for axis in ('x', 'y')}
        for axis in ('x', 'y'):
            result['v' + axis] = (finite(other[axis], 'ball ' + axis) - result[axis]) / dt
        balls.append(result)
    config.update(cars=cars, balls=balls)


def goalie_frames(seed=None):
    """A reachable, unobstructed shot aimed at the positive-Y field edge."""
    if seed is None:
        seed = random.SystemRandom().randrange(2 ** 63)
    rng = random.Random(seed)
    x, y = rng.uniform(-.55, .55), rng.uniform(-1., 0.)
    aim_x = rng.uniform(-.55, .55)
    dx, dy = aim_x - x, 1.45 - y
    scale = .8 / math.hypot(dx, dy)
    car = dict(id=1, x=0., y=1.20, angle=0.)
    return [dict(type='sim_scenario', schema=1, config=dict(field=[1.5, 2.9],
                 random_seed=seed, wait_for_control=True)),
            dict(time=0., cars=[car], balls=[dict(x=x, y=y)]),
            dict(time=.1, cars=[car], balls=[dict(x=x + .1 * dx * scale, y=y + .1 * dy * scale)])]


def seed_from_scenario(config, source, seed=None):
    try:
        if str(source) == 'goalie':
            rows = goalie_frames(seed)
        else:
            with open(source, encoding='utf-8') as stream:
                rows = []
                for line in stream:
                    if line.strip():
                        rows.append(json.loads(line))
                    if rows and rows[0].get('type') in ('bag_header', 'provider_snapshot', 'udp'):
                        seed_from_bag(config, source, motion=True)
                        return
                    if len(rows) == 3:
                        break
        if len(rows) != 3 or rows[0].get('type') != 'sim_scenario' or rows[0].get('schema') != 1:
            raise ValueError('scenario needs a schema-1 sim_scenario header and two position frames')
        overrides = rows[0].get('config', {})
        if not isinstance(overrides, dict) or overrides.keys() - config.keys() or {'cars', 'balls'} & overrides.keys():
            raise ValueError('scenario header config must contain known settings; positions belong in frames')
        for key, value in overrides.items():
            if key in ('motion', 'ball_model', 'ball_wobble', 'latency') and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = deepcopy(value)
        apply_frames(config, rows[1], rows[2])
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f'invalid scenario frames: {error}') from error


def save_scenario(config, path):
    """Write a resolved start as two positions, without overwriting an existing file."""
    header = dict(type='sim_scenario', schema=1,
                  config={k: deepcopy(v) for k, v in config.items() if k not in ('cars', 'balls')})
    frames = []
    for t in (0., .1):
        cars = [dict(id=c['id'], x=c['x'] + t * c.get('vx', 0), y=c['y'] + t * c.get('vy', 0),
                     angle=c['angle'] + t * c.get('angular_velocity_deg_s', 0)) for c in config['cars']]
        balls = [dict(x=b['x'] + t * b.get('vx', 0), y=b['y'] + t * b.get('vy', 0)) for b in config['balls']]
        frames.append(dict(time=t, cars=cars, balls=balls))
    text = '\n'.join(json.dumps(row, allow_nan=False) for row in [header, *frames]) + '\n'
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(text)
