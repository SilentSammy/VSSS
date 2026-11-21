"""Fit and validate a simple planar ball model from a VSSS observer bag.

The fitted model has two stages shared with the simulator:

* impact: the car transfers closing velocity along the car-to-ball normal;
* roll: the released ball decelerates at a constant magnitude until it stops.

Only free-roll episodes following a detected car/ball contact are used.  Long
pushes and dribbles remain visible in the diagnostics but are not treated as
ballistic kicks.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from statistics import median

from vsss._sim import BallModel, load_config


@dataclass(frozen=True)
class Sample:
    time: float
    car_x: float
    car_y: float
    ball_x: float
    ball_y: float
    car_vx: float = 0.0
    car_vy: float = 0.0
    ball_vx: float = 0.0
    ball_vy: float = 0.0


@dataclass(frozen=True)
class Roll:
    start: int
    end: int
    contact_start: int
    duration: float
    distance: float
    initial_speed: float
    final_speed: float
    car_speed: float
    closing_speed: float
    direction_x: float
    direction_y: float
    observed_end_x: float
    observed_end_y: float
    termination: str


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def load_samples(path: Path, car_id=1, field_x=1.0, field_y=1.6):
    """Load plausible, single-ball observations from an observer JSONL bag."""
    raw = []
    previous_ball = None
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get('type') != 'udp' or not event.get('valid'):
            continue
        packet = event.get('packet') or {}
        cars = [m for m in packet.get('markers', []) if m.get('id') == car_id]
        balls = packet.get('balls') or []
        if not cars or not balls:
            continue
        plausible = [b for b in balls if abs(float(b['x'])) <= field_x and
                     abs(float(b['y'])) <= field_y]
        if not plausible:
            continue
        ball = min(plausible, key=lambda b: _distance(
            (float(b['x']), float(b['y'])), previous_ball)) if previous_ball else plausible[0]
        car = cars[0]
        now = float(event['wall_time_ns']) / 1e9
        point = (float(ball['x']), float(ball['y']))
        raw.append(Sample(now, float(car['x']), float(car['y']), *point))
        previous_ball = point
    if len(raw) < 5:
        raise ValueError('bag has too few correlated car/ball observations')

    # Split at dropouts and reject isolated position jumps. Derivatives are a
    # local least-squares slope, which is substantially less noisy than a
    # frame-to-frame difference at the camera's ~10 Hz publication rate.
    clean = [raw[0]]
    for sample in raw[1:]:
        last = clean[-1]
        dt = sample.time - last.time
        jump = _distance((sample.ball_x, sample.ball_y), (last.ball_x, last.ball_y))
        if dt <= 0:
            continue
        if dt < .35 and jump / dt > 3.0:
            continue
        clean.append(sample)

    def slope(index, x_name, y_name):
        center = clean[index].time
        window = [p for p in clean[max(0, index - 2):index + 3]
                  if abs(p.time - center) <= .32]
        if len(window) < 2:
            return 0.0, 0.0
        mean_t = sum(p.time for p in window) / len(window)
        denom = sum((p.time - mean_t) ** 2 for p in window)
        if denom <= 1e-9:
            return 0.0, 0.0
        return tuple(sum((p.time - mean_t) * getattr(p, name) for p in window) / denom
                     for name in (x_name, y_name))

    result = []
    for i, p in enumerate(clean):
        cv = slope(i, 'car_x', 'car_y')
        bv = slope(i, 'ball_x', 'ball_y')
        result.append(Sample(p.time, p.car_x, p.car_y, p.ball_x, p.ball_y,
                             cv[0], cv[1], bv[0], bv[1]))
    return result


def detect_rolls(samples, contact_radius=.18, release_radius=.21,
                 minimum_distance=.10, minimum_duration=.35):
    """Find free rolls that begin after contact and end at rest/dropout/contact."""
    distances = [_distance((p.car_x, p.car_y), (p.ball_x, p.ball_y)) for p in samples]
    contact = [d <= contact_radius for d in distances]
    rolls = []
    i = 0
    armed_at = None
    while i < len(samples) - 2:
        if contact[i]:
            if armed_at is None:
                armed_at = i
            i += 1
            continue
        if armed_at is not None and distances[i] >= release_radius:
            contact_start = armed_at
            start = i
            end = start + 1
            termination = 'recording_end'
            while end + 1 < len(samples):
                gap = samples[end + 1].time - samples[end].time
                speed = math.hypot(samples[end].ball_vx, samples[end].ball_vy)
                if gap > .35:
                    termination = 'vision_gap'
                    break
                if contact[end + 1]:
                    termination = 'car_contact'
                    break
                if speed < .055:
                    termination = 'stopped'
                    break
                end += 1
            first, last = samples[start], samples[end]
            distance = _distance((first.ball_x, first.ball_y),
                                 (last.ball_x, last.ball_y))
            duration = last.time - first.time
            # Estimate launch velocity from post-release samples only.  A
            # centered derivative at the release frame includes bumper
            # contact and biases the direction sideways.
            launch_end = min(end, start + 3)
            launch = samples[launch_end]
            launch_dt = launch.time - first.time
            launch_vx = ((launch.ball_x - first.ball_x) / launch_dt
                         if launch_dt > 1e-6 else first.ball_vx)
            launch_vy = ((launch.ball_y - first.ball_y) / launch_dt
                         if launch_dt > 1e-6 else first.ball_vy)
            initial_speed = math.hypot(launch_vx, launch_vy)
            final_speed = math.hypot(last.ball_vx, last.ball_vy)
            contact_samples = samples[contact_start:start]
            car_speed = max((math.hypot(p.car_vx, p.car_vy)
                             for p in contact_samples), default=0.0)
            closing = 0.0
            for car in contact_samples:
                nx, ny = car.ball_x - car.car_x, car.ball_y - car.car_y
                norm = math.hypot(nx, ny)
                if norm:
                    closing = max(closing, (car.car_vx * nx + car.car_vy * ny) / norm)
            if duration >= minimum_duration and distance >= minimum_distance and initial_speed > .08:
                rolls.append(Roll(start, end, contact_start, duration, distance,
                                  initial_speed, final_speed, car_speed, closing,
                                  launch_vx / initial_speed, launch_vy / initial_speed,
                                  last.ball_x, last.ball_y, termination))
            armed_at = None
            i = max(i + 1, end)
            continue
        # A near-contact episode that never opens into a release is a push,
        # not an impact followed by free rolling.
        if armed_at is not None and samples[i].time - samples[armed_at].time > 1.0:
            armed_at = None
        i += 1
    return rolls


def fit_model(samples, rolls):
    if not rolls:
        raise ValueError('no clean post-contact free-roll episodes were found')
    complete = [roll for roll in rolls if roll.termination == 'stopped']
    decelerations = []
    gains = []
    for roll in complete:
        if (roll.termination == 'stopped' and roll.distance > .02 and
                roll.initial_speed > roll.final_speed):
            decelerations.append((roll.initial_speed ** 2 - roll.final_speed ** 2) /
                                 (2.0 * roll.distance))
        if roll.closing_speed > .05:
            gains.append(roll.initial_speed / roll.closing_speed)
    if not decelerations:
        raise ValueError('rolls do not contain measurable deceleration')
    # Fit constant rolling resistance plus a small field-fixed acceleration.
    # The latter captures floor/track bias that consistently bends a roll.
    rows = []
    values = []
    for roll in complete:
        start = samples[roll.start]
        t = roll.duration
        if t <= .2:
            continue
        vx, vy = (roll.initial_speed * roll.direction_x,
                  roll.initial_speed * roll.direction_y)
        dx = roll.observed_end_x - start.ball_x
        dy = roll.observed_end_y - start.ball_y
        scale = 2.0 / (t * t)
        rows.extend(([-roll.direction_x, 1.0, 0.0],
                     [-roll.direction_y, 0.0, 1.0]))
        values.extend(((dx - vx * t) * scale, (dy - vy * t) * scale))

    def solve_least_squares(matrix, target):
        normal = [[sum(row[i] * row[j] for row in matrix) for j in range(3)]
                  for i in range(3)]
        rhs = [sum(row[i] * value for row, value in zip(matrix, target))
               for i in range(3)]
        work = [normal[i] + [rhs[i]] for i in range(3)]
        for col in range(3):
            pivot = max(range(col, 3), key=lambda r: abs(work[r][col]))
            if abs(work[pivot][col]) < 1e-9:
                raise ValueError('roll directions do not constrain a planar model')
            work[col], work[pivot] = work[pivot], work[col]
            divisor = work[col][col]
            work[col] = [v / divisor for v in work[col]]
            for row in range(3):
                if row != col:
                    factor = work[row][col]
                    work[row] = [a - factor * b for a, b in zip(work[row], work[col])]
        return [row[-1] for row in work]

    if len(rows) >= 6:
        deceleration, drift_x, drift_y = solve_least_squares(rows, values)
        deceleration = max(.02, deceleration)
        # A bias at or above rolling resistance would make a resting ball move.
        drift = math.hypot(drift_x, drift_y)
        if drift >= deceleration * .8:
            scale = deceleration * .8 / drift
            drift_x, drift_y = drift_x * scale, drift_y * scale
    else:
        deceleration, drift_x, drift_y = median(decelerations), 0.0, 0.0
    return BallModel(impact_gain=median(gains) if gains else 1.0,
                     rolling_deceleration=deceleration,
                     drift_acceleration_x=drift_x,
                     drift_acceleration_y=drift_y)


def validate_leave_one_out(samples, rolls):
    """Return endpoint errors with every evaluated roll excluded from its fit."""
    results = []
    complete = [roll for roll in rolls if roll.termination == 'stopped']
    for index, roll in enumerate(complete):
        training = complete[:index] + complete[index + 1:]
        if not training:
            continue
        try:
            model = fit_model(samples, training)
        except ValueError:
            continue
        start = samples[roll.start]
        predicted = model.stopping_point(
            (start.ball_x, start.ball_y),
            (roll.initial_speed * roll.direction_x,
             roll.initial_speed * roll.direction_y))
        error = _distance(predicted, (roll.observed_end_x, roll.observed_end_y))
        predicted_launch = model.impact_gain * roll.closing_speed
        impact_predicted = model.stopping_point(
            (start.ball_x, start.ball_y),
            (predicted_launch * roll.direction_x,
             predicted_launch * roll.direction_y))
        impact_error = _distance(impact_predicted,
                                 (roll.observed_end_x, roll.observed_end_y))
        results.append({'episode': index + 1, 'predicted_end': list(predicted),
                        'observed_end': [roll.observed_end_x, roll.observed_end_y],
                        'error_m': error,
                        'observed_launch_speed_mps': roll.initial_speed,
                        'predicted_launch_speed_mps': predicted_launch,
                        'launch_speed_error_mps': abs(predicted_launch - roll.initial_speed),
                        'impact_predicted_end': list(impact_predicted),
                        'impact_endpoint_error_m': impact_error})
    return results


def analyze(path, car_id=1):
    samples = load_samples(path, car_id)
    rolls = detect_rolls(samples)
    model = fit_model(samples, rolls)
    validation = validate_leave_one_out(samples, rolls)
    errors = [row['error_m'] for row in validation]
    impact_errors = [row['impact_endpoint_error_m'] for row in validation]
    speed_errors = [row['launch_speed_error_mps'] for row in validation]
    return {'source': str(path), 'car_id': car_id, 'samples': len(samples),
            'free_roll_episodes': len(rolls), 'model': asdict(model),
            'episodes': [asdict(row) for row in rolls],
            'validation': validation,
            'validation_mae_m': sum(errors) / len(errors) if errors else None,
            'validation_median_m': median(errors) if errors else None,
            'impact_endpoint_mae_m': (sum(impact_errors) / len(impact_errors)
                                      if impact_errors else None),
            'launch_speed_mae_mps': (sum(speed_errors) / len(speed_errors)
                                     if speed_errors else None)}


def analyze_slow_rolls(path, car_id=1):
    """Describe terminal motion separately from stationary detector jitter.

    This is a diagnostic, not a new noise-free fit: camera sampling and local
    derivative smoothing cannot uniquely identify ball roughness.
    """
    samples = load_samples(path, car_id)
    rolls = detect_rolls(samples)
    quiet_steps = []
    bias = load_config()['ball_model']
    lateral = dict(fast=[], slow=[], near_stop=[])
    for a, b in zip(samples, samples[1:]):
        dt = b.time - a.time
        if not .05 < dt < .2 or min(math.hypot(p.car_x - p.ball_x, p.car_y - p.ball_y)
                                    for p in (a, b)) < .25:
            continue
        speed = math.hypot(a.ball_vx, a.ball_vy)
        if max(speed, math.hypot(b.ball_vx, b.ball_vy)) < .015:
            quiet_steps.append(math.hypot(b.ball_x - a.ball_x, b.ball_y - a.ball_y))
        if speed > .015:
            # Remove the previously fitted field-fixed bias; rolling resistance
            # is parallel to velocity, so does not enter this projection.
            ax = (b.ball_vx - a.ball_vx) / dt - bias['drift_acceleration_x']
            ay = (b.ball_vy - a.ball_vy) / dt - bias['drift_acceleration_y']
            label = 'fast' if speed >= .2 else 'slow' if speed >= .05 else 'near_stop'
            lateral[label].append(abs((-a.ball_vy * ax + a.ball_vx * ay) / speed))
    tails = []
    for index, roll in enumerate(rolls, 1):
        if roll.termination != 'stopped':
            continue
        first = samples[roll.end]
        tail = []
        previous = first
        for point in samples[roll.end:]:
            if (point.time - first.time > 1.5 or point.time - previous.time > .35 or
                    math.hypot(point.car_x - point.ball_x, point.car_y - point.ball_y) < .22):
                break
            tail.append(point)
            previous = point
        if len(tail) < 2:
            continue
        last = tail[-1]
        tails.append(dict(episode=index, duration_s=last.time - first.time,
                          initial_speed_mps=math.hypot(first.ball_vx, first.ball_vy),
                          net_displacement_m=math.hypot(last.ball_x - first.ball_x, last.ball_y - first.ball_y),
                          path_length_m=sum(math.hypot(b.ball_x - a.ball_x, b.ball_y - a.ball_y)
                                            for a, b in zip(tail, tail[1:])),
                          points=[dict(t=p.time - first.time, x=p.ball_x, y=p.ball_y) for p in tail]))
    quiet_steps.sort()
    return dict(source=str(path), samples=len(samples), free_roll_episodes=len(rolls),
                stationary_frame_steps=len(quiet_steps),
                stationary_step_median_m=median(quiet_steps) if quiet_steps else None,
                stationary_step_p95_m=quiet_steps[min(len(quiet_steps) - 1, int(.95 * len(quiet_steps)))] if quiet_steps else None,
                removed_field_drift_mps2=[bias['drift_acceleration_x'], bias['drift_acceleration_y']],
                median_absolute_lateral_acceleration_mps2={key: median(value) if value else None
                                                         for key, value in lateral.items()},
                terminal_episodes=tails,
                interpretation='Terminal motion includes physical motion and camera noise; roughness settings remain heuristic.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('--car', type=int, default=1)
    parser.add_argument('--slow-rolls', action='store_true', help='inspect terminal motion and stationary detector jitter')
    parser.add_argument('--output', type=Path,
                        help='write the fitted model and diagnostics as new JSON')
    args = parser.parse_args()
    result = analyze_slow_rolls(args.bag, args.car) if args.slow_rolls else analyze(args.bag, args.car)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        with args.output.open('x', encoding='utf-8') as destination:
            destination.write(text + '\n')


if __name__ == '__main__':
    main()
