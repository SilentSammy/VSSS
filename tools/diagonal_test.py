"""Measure guarded open-loop diagonal responses without changing robot config."""
import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from queue import SimpleQueue
import time

from vsss import Car, GPS
from tools.calibrate import response_between
from tools.calibrate_staged import StageMonitor as Monitor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--car', type=int, required=True)
    parser.add_argument('--level', type=float, default=.25)
    parser.add_argument('--seconds', type=float, default=1.2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not .1 <= args.level <= .5 or not .2 <= args.seconds <= 2.5:
        parser.error('level must be .1-.5 and duration .2-2.5 seconds')
    directory = args.output or Path('tools/runs') / time.strftime(
        '%Y%m%dT%H%M%SZ_diagonal', time.gmtime())
    directory.mkdir(parents=True, exist_ok=False)
    gps = None
    try:
        with Car(args.car, timeout=50) as car, \
                (directory / 'frames.csv').open('x', newline='', encoding='utf-8') as frames:
            car.stop()
            # Establish BLE before starting camera UDP reception, matching
            # teleop's startup order. Camera checks still precede all motion.
            gps = GPS(stale_after=.8)
            pose = car.wait_for_position(10)
            center = (pose.x, pose.y, pose.angle)
            writer = csv.writer(frames)
            writer.writerow(('elapsed_s', 'phase', 'frame_sequence', 'frame_age_ms',
                             'usable', 'x_m', 'y_m', 'heading_deg'))
            monitor = Monitor(car, gps, writer, SimpleQueue(),
                              0, center=center)
            results = []
            deadline = time.monotonic() + 30.
            try:
                monitor.wait_stopped(.5, deadline)
                commands = ((args.level, -args.level, 0.),
                            (-args.level, args.level, 0.),
                            (-args.level, -args.level, 0.),
                            (args.level, args.level, 0.))
                for index, command in enumerate(commands):
                    monitor.phase = 'baseline_%d' % index
                    start = monitor.wait_stopped(.3, deadline)
                    monitor.phase = 'diagonal_%d' % index
                    started = time.monotonic()
                    try:
                        car.set_velocity(*command)
                        monitor.wait(args.seconds, deadline, start.angle)
                    finally:
                        car.stop()
                        monitor.velocity = (0., 0.)
                        monitor.previous = None
                    end = monitor.wait_stopped(.85, deadline, start.angle)
                    response = response_between(start, end)
                    result = dict(command=command, start=asdict(start), end=asdict(end),
                                  forward_m=response[0], left_m=response[1],
                                  heading_deg=response[2], duration_s=time.monotonic()-started)
                    results.append(result)
                    print(json.dumps(result), flush=True)
            finally:
                car.stop()
                monitor.close()
            (directory / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
            print('Saved:', directory, flush=True)
    finally:
        if gps is not None:
            gps.close()


if __name__ == '__main__':
    main()
