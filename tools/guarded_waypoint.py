"""Run one monitored onboard waypoint with rectangular and predictive guards."""
import argparse
import csv
import math
from pathlib import Path
import time

from vsss.car import Car
from vsss.gps import GPS


def guarded(args):
    gps = GPS(stale_after=.35)
    output = args.output.open('x', newline='', encoding='utf-8')
    writer = csv.writer(output)
    writer.writerow(('elapsed', 'sequence', 'frame_age_ms', 'x', 'y', 'heading',
                     'predicted_x', 'predicted_y', 'mode', 'fault', 'onboard_packets',
                     'target_x', 'target_y', 'target_heading'))
    try:
        with Car(args.car, timeout=50) as car:
            try:
                pose = car.wait_for_position(timeout=10)
                status = car.read_status()
                if not status.wifi_connected or not status.pose_fresh:
                    raise RuntimeError('onboard pose is unavailable')
                if not (-.6 < pose.x < .6 and .2 < pose.y < 1.3):
                    raise RuntimeError('starting pose outside the inner guard')
                targets = [args.target] + (args.then or [])
                car.go_to(*targets[0])
                target_index = 0
                started = time.monotonic()
                waypoint_started = started
                previous = None
                vx = vy = 0.
                saw_position = False
                status_at = 0.
                while time.monotonic() - waypoint_started < args.timeout:
                    now = time.monotonic()
                    pose = car.pose
                    feedback = gps.latest
                    if pose is None or feedback is None:
                        raise RuntimeError('host pose stale or missing')
                    if previous is not None and feedback.received_at != previous[0]:
                        dt = feedback.received_at - previous[0]
                        if 0 < dt < .5:
                            vx = .5 * ((pose.x - previous[1]) / dt) + .5 * vx
                            vy = .5 * ((pose.y - previous[2]) / dt) + .5 * vy
                    if previous is None or feedback.received_at != previous[0]:
                        previous = (feedback.received_at, pose.x, pose.y)
                    px, py = pose.x + vx * .5, pose.y + vy * .5
                    if not all((-.6 < pose.x < .6, .2 < pose.y < 1.3,
                                -.6 < px < .6, .2 < py < 1.3)):
                        raise RuntimeError('inner or predicted boundary reached')
                    if now >= status_at:
                        status = car.read_status()
                        status_at = now + .3
                        if not status.wifi_connected or not status.pose_fresh:
                            raise RuntimeError('onboard pose became unavailable')
                        if status.mode_name == 'fault':
                            raise RuntimeError('onboard fault: ' + status.fault_name)
                        saw_position |= status.mode_name == 'position'
                    writer.writerow((round(now-started, 3), feedback.frame_sequence,
                                     feedback.frame_age_ms, pose.x, pose.y, pose.angle,
                                     px, py, status.mode_name, status.fault_name, status.packet_count,
                                     *targets[target_index]))
                    output.flush()
                    if saw_position and status.mode_name == 'reached':
                        print('reached', pose, 'seconds', round(now-started, 2), flush=True)
                        target_index += 1
                        if target_index == len(targets):
                            return
                        car.go_to(*targets[target_index])
                        waypoint_started = time.monotonic()
                        saw_position = False
                        status_at = 0.
                    time.sleep(.05)
                raise TimeoutError('waypoint observation timed out')
            finally:
                car.stop()
    finally:
        output.close()
        gps.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', nargs=3, type=float, metavar=('X', 'Y', 'ANGLE'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--car', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=15.)
    parser.add_argument('--then', action='append', type=float, nargs=3,
                        metavar=('X', 'Y', 'ANGLE'), help='additional guarded waypoint; may be repeated')
    args = parser.parse_args()
    if not (all(all(math.isfinite(v) for v in target) and
                -.4 <= target[0] <= .4 and .4 <= target[1] <= 1.1
                for target in [args.target] + (args.then or [])) and
            0 < args.timeout <= 30):
        parser.error('target or timeout outside guarded range')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    guarded(args)


if __name__ == '__main__':
    main()
