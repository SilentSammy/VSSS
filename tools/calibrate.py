"""Measure, apply, or reset a robot's trim calibration.

The default motion region is centered on the field. Run only with
the marker attached and a clear, supervised track. Connection/preflight time is
outside the 30-second motion budget. Interrupted runs keep their recordings.
"""
import argparse
import csv
from dataclasses import asdict
import json
import math
import sys
from pathlib import Path
from queue import Empty, SimpleQueue
import time

from vsss.car import Car
from vsss.gps import GPS
from vsss._ble_client import normalize_trim_matrix
from ._fit_trim import fit


PULSE_SECONDS = .8
SETTLE_SECONDS = .85
MOTION_BUDGET = 30.
INNER_X = .60
INNER_Y = 1.30
LOCAL_Y_RADIUS = .55
CENTER = (0., 0., 90.)
YAW_SECONDS = .70
MAX_PULSE_ROTATION_DEG = 70.


class PoseUnavailable(RuntimeError):
    """The host does not currently have a usable, fresh robot pose."""


class CalibrationQualityError(RuntimeError):
    """Measurements produced a proposal that must not be applied."""


def angle_delta(a, b):
    return (a - b + 180.) % 360. - 180.


def response_between(start, end):
    theta = math.radians(start.angle)
    dx, dy = end.x - start.x, end.y - start.y
    return (dx * math.cos(theta) + dy * math.sin(theta),
            -dx * math.sin(theta) + dy * math.cos(theta),
            angle_delta(end.angle, start.angle))


def assess(rows, result, original):
    """Reject weak, asymmetric, ill-conditioned, or unusually large changes."""
    errors = []
    for axis, key in enumerate(('forward_m', 'left_m', 'heading_deg')):
        pair = [row for row in rows if row['axis'] == axis]
        minimum = 2. if axis == 2 else .01
        gains = [row[key] / row['command'] for row in pair]
        if len(pair) != 2 or any(row['frames'] < 3 for row in pair):
            errors.append('axis %d has too few fresh pulse frames' % axis)
        if len(pair) != 2 or any(abs(row[key]) < minimum for row in pair):
            errors.append('axis %d displacement is too small for a reliable fit' % axis)
        if any(abs(row['heading_deg']) > MAX_PULSE_ROTATION_DEG for row in pair):
            errors.append('axis %d rotation exceeded the 70-degree guard' % axis)
        if len(gains) == 2 and (min(gains) <= 0 or max(gains) > 3 * min(gains)):
            errors.append('axis %d responses have wrong signs or disagree by over 3x' % axis)
    response = result['response_matrix']
    if any(abs(response[i][i]) < 1e-8 for i in range(3)):
        errors.append('an axis has insufficient main response')
    else:
        # Normalize output units before measuring inverse conditioning.
        normalized = [[response[r][c] / response[r][r] for c in range(3)] for r in range(3)]
        from ._fit_trim import inverse
        inv = inverse(normalized)
        condition = max(sum(abs(v) for v in row) for row in normalized) * max(
            sum(abs(v) for v in row) for row in inv)
        if condition > 5:
            errors.append('response matrix is poorly conditioned (%.1f)' % condition)
    change, r, c = max((abs(result['matrix'][r][c] - original[r][c]), r, c)
                       for r in range(3) for c in range(3))
    if change > .15:
        errors.append('proposed matrix change %.3f exceeds 0.15 at row %d, column %d (%.3f to %.3f)' %
                      (change, r + 1, c + 1, original[r][c], result['matrix'][r][c]))
    return errors


class Monitor:
    def __init__(self, car, gps, writer, events, initial_packets, center=CENTER):
        self.car, self.gps, self.writer, self.events = car, gps, writer, events
        self.center = center
        self.previous = None
        self.velocity = (0., 0.)
        self.phase = 'preflight'
        self.last_robot_packets = initial_packets
        self.ble_generation = getattr(car, 'connection_generation', None)
        self.started = time.monotonic()
        self.frames = []
        self.gps.add_listener(self.on_feedback)

    def on_feedback(self, feedback):
        self.events.put(feedback)

    def check_robot_status(self):
        status = self.car.read_status()
        if status.packet_count < self.last_robot_packets:
            raise RuntimeError('robot telemetry counter reset; possible restart or reconnect')
        self.last_robot_packets = status.packet_count
        if not status.wifi_connected or not status.pose_fresh or status.mode_name == 'fault':
            raise RuntimeError('onboard telemetry unavailable or controller fault')
        return status

    def drain(self):
        while True:
            try:
                feedback = self.events.get_nowait()
            except Empty:
                break
            pose = feedback.cars.get(self.car.number)
            self.frames.append((feedback.received_at, feedback.frame_sequence, pose, feedback.usable))
            self.writer.writerow((round(feedback.received_at-self.started, 4), self.phase,
                                  feedback.frame_sequence, feedback.frame_age_ms, feedback.usable,
                                  '' if pose is None else pose.x, '' if pose is None else pose.y,
                                  '' if pose is None else pose.angle))

    def check(self, deadline=None):
        self.drain()
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('30-second motion budget exceeded')
        if not self.car.is_connected:
            raise RuntimeError('BLE disconnected')
        if (self.ble_generation is not None and
                self.car.connection_generation != self.ble_generation):
            raise RuntimeError('BLE session reconnected; calibration run invalid')
        pose, feedback = self.car.pose, self.gps.latest
        if pose is None or feedback is None:
            raise PoseUnavailable('fresh host pose unavailable')
        if self.previous and feedback.received_at != self.previous[0]:
            dt = feedback.received_at - self.previous[0]
            if 0 < dt < .5:
                self.velocity = tuple(.5 * old + .5 * new for old, new in zip(
                    self.velocity, ((pose.x-self.previous[1])/dt, (pose.y-self.previous[2])/dt)))
        if not self.previous or feedback.received_at != self.previous[0]:
            self.previous = (feedback.received_at, pose.x, pose.y)
        px, py = pose.x + .5*self.velocity[0], pose.y + .5*self.velocity[1]
        min_x, max_x = max(-INNER_X, self.center[0] - INNER_X), min(INNER_X, self.center[0] + INNER_X)
        min_y, max_y = max(-INNER_Y, self.center[1] - LOCAL_Y_RADIUS), min(INNER_Y, self.center[1] + LOCAL_Y_RADIUS)
        if not (min_x < pose.x < max_x and min_y < pose.y < max_y and
                min_x < px < max_x and min_y < py < max_y):
            raise RuntimeError('actual or predicted position reached the inner boundary')
        return pose

    def wait(self, seconds, deadline, starting_heading=None):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            pose = self.check(deadline)
            if (starting_heading is not None and
                    abs(angle_delta(pose.angle, starting_heading)) > MAX_PULSE_ROTATION_DEG):
                raise RuntimeError('pulse rotation exceeded the 70-degree guard')
            time.sleep(.04)
        pose = self.check(deadline)
        if (starting_heading is not None and
                abs(angle_delta(pose.angle, starting_heading)) > MAX_PULSE_ROTATION_DEG):
            raise RuntimeError('pulse rotation exceeded the 70-degree guard')
        return pose

    def wait_stopped(self, seconds, deadline, starting_heading=None):
        """Wait for continuous fresh feedback while already stopped.

        A brief gap may be reacquired here; active movement always aborts on
        the first missing pose. The overall motion deadline still applies.
        """
        end = time.monotonic() + seconds
        recovery_deadline = None
        while True:
            try:
                pose = self.check(deadline)
            except PoseUnavailable:
                if recovery_deadline is None:
                    recovery_deadline = time.monotonic() + 3.
                    print('telemetry unavailable while stopped; waiting for recovery', flush=True)
                if time.monotonic() >= recovery_deadline:
                    raise
                end = time.monotonic() + seconds
                time.sleep(.04)
                continue
            if (starting_heading is not None and
                    abs(angle_delta(pose.angle, starting_heading)) > MAX_PULSE_ROTATION_DEG):
                raise RuntimeError('pulse rotation exceeded the 70-degree guard during settling')
            if time.monotonic() >= end:
                return pose
            time.sleep(.04)

    def close(self):
        self.gps.remove_listener(self.on_feedback)
        self.drain()


def recenter(monitor, deadline):
    """Use onboard control and require a new position-mode observation."""
    car = monitor.car
    monitor.phase = 'recenter'
    monitor.check(deadline)
    car.go_to(*monitor.center)
    saw_position = False
    until = min(deadline, time.monotonic() + 7.)
    next_status = 0.
    try:
        while time.monotonic() < until:
            monitor.check(deadline)
            now = time.monotonic()
            if now >= next_status:
                status = monitor.check_robot_status()
                next_status = time.monotonic() + .35
                saw_position |= status.mode_name == 'position'
                if saw_position and status.mode_name == 'reached':
                    return
            time.sleep(.04)
        raise TimeoutError('onboard recenter did not complete')
    finally:
        car.stop()


def calibrate(car, gps, directory, auto_apply=False, *, translation_level=.50,
              yaw_level=.20, pulse_seconds=PULSE_SECONDS, yaw_seconds=YAW_SECONDS,
              ask_apply=False, save_json=None, center=CENTER):
    if yaw_seconds is None:
        yaw_seconds = pulse_seconds
    original_trim = car.read_trim()
    original_controller = car.read_controller()
    status = car.read_status()
    if not status.wifi_connected or not status.pose_fresh or status.mode_name != 'idle':
        raise RuntimeError('robot must be idle with Wi-Fi and fresh onboard pose')
    with (directory / 'original.json').open('x', encoding='utf-8') as destination:
        json.dump({'trim': original_trim, 'controller': original_controller,
                   'status': asdict(status), 'pulse_seconds': pulse_seconds,
                   'translation_level': translation_level, 'yaw_level': yaw_level,
                   'yaw_seconds': yaw_seconds, 'center': center},
                  destination, indent=2)
    events = SimpleQueue()
    with (directory / 'frames.csv').open('x', newline='', encoding='utf-8') as recording, \
            (directory / 'pulses.csv').open('x', newline='', encoding='utf-8') as pulse_file:
        writer = csv.writer(recording)
        writer.writerow(('elapsed_s', 'phase', 'frame_sequence', 'frame_age_ms',
                         'usable', 'x_m', 'y_m', 'heading_deg'))
        pulse_writer = csv.DictWriter(pulse_file, fieldnames=(
            'axis', 'command', 'command_x', 'command_y', 'command_w',
            'forward_m', 'left_m', 'heading_deg', 'frames', 'duration_s'))
        pulse_writer.writeheader()
        pulse_file.flush()
        monitor = Monitor(car, gps, writer, events, status.packet_count, center)
        try:
            pose = car.wait_for_position(timeout=10)
            if math.hypot(pose.x-status.x, pose.y-status.y) > .15 or abs(angle_delta(pose.angle, status.angle)) > 20:
                raise RuntimeError('host and onboard poses disagree')
            # Give the camera time to settle before the first burst.
            monitor.wait_stopped(.5, None)
            deadline = time.monotonic() + MOTION_BUDGET
            pose = monitor.check(deadline)
            if math.hypot(pose.x-center[0], pose.y-center[1]) > .08 or abs(angle_delta(pose.angle, center[2])) > 8:
                print('recentering before pulses', flush=True)
                recenter(monitor, deadline)
            rows = []
            for axis in range(3):
                monitor.check_robot_status()
                for sign in (1, -1):
                    for attempt in range(3):
                        monitor.phase = 'baseline'
                        start_pose = monitor.wait_stopped(.25, deadline)
                        command = [0., 0., 0.]
                        command[axis] = sign * (yaw_level if axis == 2 else translation_level)
                        monitor.phase = 'pulse_%d_%+d' % (axis, sign)
                        pulse_started = time.monotonic()
                        interrupted = False
                        try:
                            car.set_velocity(*command)
                            duration = yaw_seconds if axis == 2 else pulse_seconds
                            monitor.wait(duration, deadline, start_pose.angle)
                        except PoseUnavailable:
                            interrupted = True
                        finally:
                            car.stop()
                        stopped_at = time.monotonic()
                        if not interrupted:
                            break
                        print('pulse interrupted by telemetry loss; stopped, discarding attempt', flush=True)
                        monitor.phase = 'interrupted_settle'
                        if attempt == 2:
                            raise PoseUnavailable('active pulse lost telemetry on three attempts')
                        pose = monitor.wait_stopped(SETTLE_SECONDS, deadline, start_pose.angle)
                        if math.hypot(pose.x-center[0], pose.y-center[1]) > .12:
                            recenter(monitor, deadline)
                    monitor.phase = 'settle'
                    end_pose = monitor.wait_stopped(SETTLE_SECONDS, deadline, start_pose.angle)
                    samples = {seq for received, seq, pose, usable in monitor.frames
                               if pulse_started <= received <= stopped_at and pose is not None and usable and seq is not None}
                    forward, left, heading = response_between(start_pose, end_pose)
                    row = dict(axis=axis, command=command[axis], command_x=command[0],
                               command_y=command[1], command_w=command[2],
                               forward_m=forward, left_m=left, heading_deg=heading,
                               frames=len(samples), duration_s=duration)
                    rows.append(row)
                    pulse_writer.writerow(row)
                    pulse_file.flush()
                    print('pulse', axis, sign, 'response', (round(forward, 4), round(left, 4), round(heading, 2)),
                          'frames', len(samples), flush=True)
                # Counter-pulses ordinarily return close to center; recenter if not.
                pose = monitor.check(deadline)
                if math.hypot(pose.x-center[0], pose.y-center[1]) > .08 or abs(angle_delta(pose.angle, center[2])) > 8:
                    print('recentering', flush=True)
                    recenter(monitor, deadline)
            monitor.phase = 'complete'
            car.stop()
            monitor.check_robot_status()
            result = fit(rows, original_trim)
            errors = assess(rows, result, original_trim)
            result.update(quality_errors=errors, controller_proposal=None,
                          controller_note='Short pulses do not establish safe closed-loop gains; original retained.',
                          motion_seconds=round(MOTION_BUDGET - max(0., deadline-time.monotonic()), 2))
            with (directory / 'proposal.json').open('x', encoding='utf-8') as destination:
                json.dump(result, destination, indent=2)
            print(json.dumps(result, indent=2), flush=True)
            if save_json is not None:
                with save_json.open('x', encoding='utf-8') as destination:
                    json.dump(result, destination, indent=2)
                print('Saved proposal:', save_json, flush=True)
            if errors:
                raise CalibrationQualityError('calibration quality checks failed: ' + '; '.join(errors))
            if ask_apply:
                try:
                    auto_apply = input('Apply this trim to the car? [y/N] ').strip().lower() in ('y', 'yes')
                except EOFError:
                    print('No answer; proposal not applied.', flush=True)
            if auto_apply:
                # Write only after saving the proposal and original for rollback.
                monitor.check()
                status = monitor.check_robot_status()
                if not status.wifi_connected or not status.pose_fresh or status.mode_name != 'idle':
                    raise RuntimeError('robot not ready for applying the proposal')
                applied = car.set_trim(result['matrix'])
                with (directory / 'applied.json').open('x', encoding='utf-8') as destination:
                    json.dump({'trim': applied, 'controller': original_controller}, destination, indent=2)
                print('Proposed trim applied and read back.', flush=True)
            return result
        finally:
            try:
                car.stop()
            finally:
                monitor.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--car', type=int, required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--auto-apply', action='store_true', help='apply a proposal that passes quality checks')
    action.add_argument('--ask-apply', action='store_true', help='ask whether to apply the measured proposal')
    action.add_argument('--apply', '--restore-trim', dest='apply', type=Path, metavar='JSON',
                        help='apply a saved proposal or original trim without moving')
    action.add_argument('--reset', action='store_true', help='reset trim to the identity matrix without moving')
    parser.add_argument('--save-json', type=Path, metavar='JSON', help='also save the proposal to this new file')
    parser.add_argument('--translation-level', type=float, default=.50,
                        help='signed x/y pulse magnitude in .05 steps (default: .50, max: .70)')
    parser.add_argument('--yaw-level', type=float, default=.20,
                        help='signed yaw pulse magnitude in .05 steps (default: .20, max: .70)')
    parser.add_argument('--pulse-seconds', type=float, default=PULSE_SECONDS,
                        help='translation pulse duration (default: .8, max: 1.2 seconds)')
    parser.add_argument('--yaw-seconds', type=float, default=YAW_SECONDS,
                        help='yaw pulse duration (default: .70 seconds)')
    parser.add_argument('--center', type=float, nargs=3, metavar=('X', 'Y', 'ANGLE'), default=CENTER,
                        help='calibration center in meters and degrees (default: 0 0 90)')
    parser.add_argument('--output', type=Path, help='new run directory (default: tools/runs/timestamp)')
    args = parser.parse_args(argv)
    if args.car < 0:
        parser.error('car id must be nonnegative')
    if (not all(math.isfinite(level) and .1 <= level <= .7 and
                abs(level * 20 - round(level * 20)) < 1e-8
                for level in (args.translation_level, args.yaw_level)) or
            not math.isfinite(args.pulse_seconds) or not .1 <= args.pulse_seconds <= 1.2):
        parser.error('pulse levels must be .10 to .70 in .05 steps and duration .10 to 1.20 seconds')
    if args.yaw_seconds is not None and (not math.isfinite(args.yaw_seconds) or not .1 <= args.yaw_seconds <= 1.2):
        parser.error('yaw duration must be .10 to 1.20 seconds')
    if not all(math.isfinite(v) for v in args.center) or not (-INNER_X < args.center[0] < INNER_X and -INNER_Y < args.center[1] < INNER_Y):
        parser.error('center must be finite and inside the field safety margins')
    if args.apply or args.reset:
        if args.output or args.save_json:
            parser.error('--apply and --reset cannot be combined with --output or --save-json')
        saved = ((1, 0, 0), (0, 1, 0), (0, 0, 1)) if args.reset else json.loads(args.apply.read_text(encoding='utf-8'))
        if isinstance(saved, dict):
            if saved.get('quality_errors'):
                parser.error('saved proposal failed quality checks')
            saved = saved.get('trim', saved.get('matrix'))
        if saved is None:
            parser.error('restore file must contain a trim matrix')
        matrix = normalize_trim_matrix(saved)
        gps = GPS()
        try:
            with Car(args.car, timeout=50) as car:
                car.stop()
                actual = car.set_trim(matrix)
                print('Trim applied and read back:', json.dumps(actual), flush=True)
        finally:
            gps.close()
        return
    directory = args.output or Path(__file__).with_name('runs') / time.strftime('%Y%m%dT%H%M%SZ_calibration', time.gmtime())
    directory.mkdir(parents=True, exist_ok=False)
    print('Recording:', directory.resolve(), flush=True)
    gps = GPS(stale_after=.35)
    try:
        with Car(args.car, timeout=50) as car:
            try:
                calibrate(car, gps, directory, args.auto_apply,
                          translation_level=args.translation_level, yaw_level=args.yaw_level,
                          pulse_seconds=args.pulse_seconds, yaw_seconds=args.yaw_seconds,
                          ask_apply=args.ask_apply, save_json=args.save_json, center=args.center)
            except CalibrationQualityError as error:
                print('\n%s\nNo trim was applied. Measurements are saved in %s. '
                      'Check the track and repeat the calibration; avoid applying a failed proposal.'
                      % (error, directory.resolve()), file=sys.stderr, flush=True)
                raise SystemExit(1) from None
    finally:
        gps.close()


if __name__ == '__main__':
    main()
