"""Offline moving-lookahead experiment using the unmodified onboard controller.

Run: python -m tools.polygon_follow_sim
Never connects to a provider or live hardware.
"""
import csv
import json
import math
from collections import deque
from pathlib import Path

from vsss._sim import SimWorld, load_config


RADIUS = .5
POINTS = [(RADIUS * math.cos(i * math.tau / 12),
           RADIUS * math.sin(i * math.tau / 12)) for i in range(13)]
EDGE = math.dist(POINTS[0], POINTS[1])
LENGTH = 12 * EDGE


def point_at(progress):
    index = min(len(POINTS) - 2, int(progress / EDGE))
    fraction = min(1., (progress - index * EDGE) / EDGE)
    a, b = POINTS[index:index + 2]
    return tuple(a[j] + fraction * (b[j] - a[j]) for j in range(2))


def project(position, progress):
    # Only search nearby forward segments: crossing the start cannot skip a lap.
    first = min(len(POINTS) - 2, int(progress / EDGE))
    candidates = []
    search = max(3, math.ceil(.4 / EDGE))
    for index in range(max(0, first - search), min(len(POINTS) - 1, first + search)):
        a, b = POINTS[index:index + 2]
        dx, dy = b[0] - a[0], b[1] - a[1]
        fraction = max(0., min(1., ((position[0] - a[0]) * dx +
                                  (position[1] - a[1]) * dy) / EDGE ** 2))
        nearest = (a[0] + fraction * dx, a[1] + fraction * dy)
        candidates.append((math.dist(position, nearest), (index + fraction) * EDGE))
    error, along = min(candidates)
    return max(progress, along), error


def trial(lookahead, update_hz):
    config = load_config()
    config.update(cars=[dict(id=1, x=POINTS[0][0], y=POINTS[0][1], angle=0.)], balls=[],
                  wait_for_control=False, random_seed=1)
    world = SimWorld(config)
    car = world.cars[1]
    pending = deque()
    progress = next_send = 0.
    final_sent = False
    rows = []
    for step in range(120 * 200):
        now = world.time
        pose = world.feedback.get(1)
        if pose is not None and now + 1e-9 >= next_send and not final_sent:
            progress, _ = project(pose[:2], progress)
            target_progress = min(LENGTH, progress + lookahead)
            target = (*point_at(target_progress), 0.)
            pending.append((now + world.ble_latency, target))
            next_send = now + 1 / update_hz
            final_sent = target_progress >= LENGTH
        while pending and pending[0][0] <= now + 1e-9:
            world.waypoint(car, pending.popleft()[1])
        world.step(1 / 200)
        _, error = project((car.x, car.y), progress)
        if step % 4 == 0:
            rows.append((world.time, car.x, car.y, error,
                         math.hypot(car.vx, car.vy), progress))
        if car.mode in (3, 4):
            break
    errors = [row[3] for row in rows]
    result = dict(lookahead_m=lookahead, update_hz=update_hz,
                  completed=car.mode == 3, duration_s=world.time,
                  rms_error_m=math.sqrt(sum(e * e for e in errors) / len(errors)),
                  max_error_m=max(errors), endpoint_error_m=math.dist((car.x, car.y), POINTS[-1]),
                  mode=car.mode, fault=car.fault,
                  telemetry_hz=world.telemetry_hz,
                  udp_latency_ms=world.udp_latency * 1000,
                  ble_latency_ms=world.ble_latency * 1000)
    return result, rows


def main():
    output = Path('artifacts/polygon_follow')
    output.mkdir(parents=True, exist_ok=True)
    trials = [trial(distance, rate) for distance in (.08, .15, .25) for rate in (5, 10)]
    results = [result for result, _ in trials]
    (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    for axis, (result, rows) in zip(axes.flat, trials):
        label = '%g m / %g Hz' % (result['lookahead_m'], result['update_hz'])
        axis.plot(*zip(*POINTS), '--', color='gray', label='Polygon')
        axis.plot([r[1] for r in rows], [r[2] for r in rows], label='Car')
        axis.set_title(label + '\nRMS %.1f cm; %s' % (
            result['rms_error_m'] * 100, 'completed' if result['completed'] else 'incomplete'))
        axis.set_aspect('equal')
        axis.grid(alpha=.3)
        with (output / ('trace_%g_%g.csv' % (result['lookahead_m'], result['update_hz']))).open('w', newline='') as target:
            writer = csv.writer(target)
            writer.writerow(('time_s', 'x_m', 'y_m', 'cross_track_m', 'speed_m_s', 'progress_m'))
            writer.writerows(rows)
    axes[0, 0].legend()
    figure.suptitle('Moving lookahead: regular dodecagon, fixed heading, current waypoint controller')
    figure.tight_layout()
    figure.savefig(output / 'paths.png', dpi=150)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
