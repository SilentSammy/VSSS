"""Repeat onboard waypoints across restarts, recording UDP/BLE and trial notes.

Preview is the default. --execute enables motion; Ctrl+C stops the experiment.
No trim/controller settings are written. Recordings use vsss-observer's JSONL
bag format, with separate timestamped trial notes and a final summary.
"""
import argparse
from collections import deque
from dataclasses import asdict
import json
import math
from pathlib import Path
from threading import Event, Lock, Thread
import time

from vsss import Car, GPS
from tools.vsss_observer import main as observe


def oscillating(samples, target):
    """Flag repeated crossings outside a noise band in the last six seconds."""
    if len(samples) < 10 or samples[-1][0] - samples[0][0] < 4:
        return False
    for axis, band in ((1, .025), (2, .025), (3, 5.)):
        signs = []
        for sample in samples:
            error = sample[axis] - target[axis - 1]
            if axis == 3:
                error = (error + 180) % 360 - 180
            if abs(error) > band:
                sign = 1 if error > 0 else -1
                if not signs or signs[-1] != sign:
                    signs.append(sign)
        if len(signs) >= 7:
            return True
    return False


class Guard:
    """Independent GPS watchdog, also active during blocking BLE operations."""
    def __init__(self, car, gps, recorder_errors, note):
        self.car, self.gps = car, gps
        self.recorder_errors, self.note = recorder_errors, note
        self.done, self.aborted = Event(), Event()
        self.paused = Event()
        self.reason = None
        self.thread = Thread(target=self.watch, daemon=True)

    def watch(self):
        previous = None
        vx = vy = 0.
        while not self.done.wait(.05):
            if self.paused.is_set():
                previous = None
                vx = vy = 0.
                continue
            pose, frame = self.car.pose, self.gps.latest
            reason = None
            if self.recorder_errors:
                reason = 'recording failed: ' + str(self.recorder_errors[0])
            elif pose is None or frame is None:
                reason = 'desktop pose missing/stale'
            else:
                if previous and frame.received_at != previous[0]:
                    dt = frame.received_at - previous[0]
                    if 0 < dt < .5:
                        vx = .5 * vx + .5 * (pose.x - previous[1]) / dt
                        vy = .5 * vy + .5 * (pose.y - previous[2]) / dt
                previous = (frame.received_at, pose.x, pose.y)
                if (abs(pose.x) >= .6 or abs(pose.y) >= 1.0 or
                        abs(pose.x + vx * .35) >= .6 or abs(pose.y + vy * .35) >= 1.0):
                    reason = 'actual/predicted safety boundary reached'
            if reason:
                self.reason = reason
                self.aborted.set()
                self.note('guard_stop', reason=reason)
                try:
                    self.car.stop()
                except Exception as error:
                    self.note('stop_error', error=str(error))
                return

    def check(self):
        if self.aborted.is_set():
            raise RuntimeError(self.reason)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--car', type=int, default=1)
    parser.add_argument('--batches', type=int, nargs='+', default=[5, 10, 15, 20],
                        help='round trips per batch, with a reboot between batches')
    parser.add_argument('--timeout', type=float, default=25., help='seconds allowed per waypoint')
    parser.add_argument('--max-seconds', type=float, default=1500., help='whole-run time limit')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--restart-first', action='store_true', help='also reboot before the first batch')
    args = parser.parse_args(argv)
    if (args.car < 0 or any(n <= 0 for n in args.batches) or
            not all(math.isfinite(v) and v > 0 for v in (args.timeout, args.max_seconds))):
        parser.error('car must be nonnegative; batch sizes and time limits must be positive')
    targets = ((0., .4, 90.), (0., -.4, 90.))
    if not args.execute:
        print(json.dumps(dict(car=args.car, targets=targets, round_trips=args.batches,
                              bounds=dict(x=[-.6, .6], y=[-1., 1.]), execute=False), indent=2))
        return
    directory = args.output or Path('tools/runs') / time.strftime('%Y%m%dT%H%M%SZ_waypoint_repeat', time.gmtime())
    directory.mkdir(parents=True, exist_ok=False)
    record_stop, record_ready = Event(), Event()
    record_errors = []
    result = dict(completed_legs=0, completed_batches=0, outcome='running', output=str(directory.resolve()))
    started = time.monotonic()
    car = gps = guard = None

    def recorder(topics, path, stop, ready):
        try:
            observe(['--topics', topics, '--quiet', '--output', str(path)],
                    stop_event=stop, ready_event=ready)
        except BaseException as error:
            record_errors.append(error)

    recording = Thread(target=recorder, args=('ble', directory / 'ble.jsonl', record_stop, record_ready),
                       name='waypoint-ble-bag', daemon=True)
    recording.start()
    udp_thread = udp_stop = None

    def start_udp(batch):
        nonlocal udp_thread, udp_stop
        udp_stop, ready = Event(), Event()
        udp_thread = Thread(target=recorder,
                            args=('udp', directory / f'udp_batch_{batch:02d}.jsonl', udp_stop, ready),
                            name='waypoint-udp-bag', daemon=True)
        udp_thread.start()
        deadline = time.monotonic() + 12
        while not ready.wait(.1):
            if record_errors or time.monotonic() > deadline:
                raise RuntimeError('UDP recording did not start: ' + str(record_errors))

    def stop_udp():
        if udp_stop:
            udp_stop.set()
            udp_thread.join(timeout=10)
            if udp_thread.is_alive():
                raise RuntimeError('UDP recording did not stop')
    with (directory / 'trials.jsonl').open('x', encoding='utf-8') as notes:
        note_lock = Lock()

        def note(kind, **values):
            event = dict(type=kind, monotonic_ns=time.monotonic_ns(), wall_time_ns=time.time_ns(), **values)
            with note_lock:
                notes.write(json.dumps(event) + '\n')
                notes.flush()
            if kind != 'sample':
                print(json.dumps(event), flush=True)

        try:
            while not record_ready.wait(.1):
                if record_errors or time.monotonic() - started > 12:
                    raise RuntimeError('recorder did not start: ' + str(record_errors))
            car = Car(args.car, timeout=25)
            # Keep BLE recording continuous. Delay Wi-Fi subscriptions until
            # BLE connects: some lab adapters cannot discover BLE while UDP
            # reception is active. During reboots the car is explicitly stopped
            # and UDP is paused; each such gap is noted below.
            if args.restart_first:
                note('restart_start', batch=1)
                car.restart(timeout=50)
                note('restart_complete', batch=1, status=asdict(car.read_status()))
            start_udp(1)
            gps = GPS()
            car.wait_for_position(timeout=15)
            guard = Guard(car, gps, record_errors, note)
            guard.thread.start()
            note('configuration', trim=car.read_trim(), controller=car.read_controller(),
                 batches=args.batches, targets=targets)
            for batch, rounds in enumerate(args.batches, 1):
                if batch > 1:
                    guard.check()
                    car.stop()
                    guard.paused.set()
                    gps.close()
                    stop_udp()
                    note('udp_pause_for_ble_discovery', batch=batch, car_stopped=True)
                    note('restart_start', batch=batch)
                    car.restart(timeout=50)
                    recovery = time.monotonic() + 25
                    while True:
                        guard.check()
                        status = car.read_status()
                        if status.pose_fresh and status.wifi_connected:
                            break
                        if time.monotonic() > recovery:
                            raise TimeoutError('onboard telemetry did not recover after reboot')
                        time.sleep(.25)
                    note('restart_complete', batch=batch, status=asdict(status),
                         trim=car.read_trim(), controller=car.read_controller())
                    start_udp(batch)
                    gps = GPS()
                    car.wait_for_position(timeout=15)
                    guard.gps = gps
                    guard.paused.clear()
                    note('udp_resume', batch=batch)
                note('batch_start', batch=batch, round_trips=rounds)
                for leg in range(rounds * 2):
                    guard.check()
                    if time.monotonic() - started > args.max_seconds:
                        raise TimeoutError('whole-run time limit')
                    target = targets[leg % 2]
                    status = car.read_status()
                    if not status.pose_fresh or not status.wifi_connected or status.fault:
                        raise RuntimeError('onboard feedback unavailable before waypoint')
                    history = deque()
                    previous_frame = None
                    leg_started = time.monotonic()
                    note('waypoint_start', batch=batch, leg=leg + 1, target=target, pose=asdict(car.pose))
                    car.go_to(*target)
                    reached_at = None
                    while True:
                        guard.check()
                        now = time.monotonic()
                        if now - leg_started > args.timeout:
                            raise TimeoutError('waypoint did not settle before timeout')
                        pose, frame = car.pose, gps.latest
                        if pose is None or frame is None:
                            raise RuntimeError('pose unavailable during waypoint')
                        if frame.received_at != previous_frame:
                            history.append((now, pose.x, pose.y, pose.angle))
                            previous_frame = frame.received_at
                        while history and history[0][0] < now - 6:
                            history.popleft()
                        status = car.read_status()
                        note('sample', batch=batch, leg=leg + 1, target=target,
                             pose=asdict(pose), status=asdict(status), frame_sequence=frame.frame_sequence,
                             frame_age_ms=frame.frame_age_ms)
                        if status.fault or not status.pose_fresh or not status.wifi_connected:
                            raise RuntimeError('onboard feedback/control fault: ' + status.fault_name)
                        if oscillating(history, target):
                            note('oscillation_detected', batch=batch, leg=leg + 1, target=target)
                            raise RuntimeError('sustained waypoint oscillation detected')
                        error = math.hypot(pose.x - target[0], pose.y - target[1])
                        heading_error = abs((pose.angle - target[2] + 180) % 360 - 180)
                        if status.mode_name == 'reached':
                            reached_at = now if reached_at is None else reached_at
                            if now - reached_at >= 1.:
                                if error > .07 or heading_error > 12:
                                    raise RuntimeError('firmware reported reached but desktop pose is off target')
                                result['completed_legs'] += 1
                                note('waypoint_complete', batch=batch, leg=leg + 1,
                                     elapsed_s=now - leg_started, error_m=error, heading_error_deg=heading_error)
                                break
                        time.sleep(.1)
                result['completed_batches'] += 1
                car.stop()
                note('batch_complete', batch=batch)
            result['outcome'] = 'completed_without_detected_glitch'
        except BaseException as error:
            result.update(outcome='stopped', reason=str(error) or type(error).__name__)
            note('experiment_stopped', reason=result['reason'])
        finally:
            if car:
                try:
                    car.stop()
                except Exception as error:
                    note('stop_error', error=str(error))
            if guard:
                guard.done.set()
                guard.thread.join(timeout=3)
            try:
                Car.close_all()
            except Exception as error:
                note('cleanup_error', error=str(error))
            if gps:
                gps.close()
            try:
                stop_udp()
            except Exception as error:
                result['recording_error'] = str(error)
            time.sleep(.5)  # Preserve final stop/disconnect events in the bag.
            record_stop.set()
            recording.join(timeout=10)
            if recording.is_alive() or record_errors:
                result['recording_error'] = str(record_errors) or 'recording thread did not finish'
            result['elapsed_s'] = time.monotonic() - started
            note('experiment_end', **result)
    (directory / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return 0 if result['outcome'] == 'completed_without_detected_glitch' else 1


if __name__ == '__main__':
    raise SystemExit(main())
