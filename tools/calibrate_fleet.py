"""Sequential live fleet calibration with camera clearance guards and return trips."""
import json
import math
import argparse
from pathlib import Path
import time

from vsss import Car, GPS
from tools import calibrate as calibration


class FleetMonitor(calibration.Monitor):
    def check(self, deadline=None):
        pose = super().check(deadline)
        feedback = self.gps.latest
        if feedback is None or not feedback.usable:
            raise RuntimeError('fleet camera feedback unavailable')
        for number in range(1, 5):
            if number == self.car.number:
                continue
            other = feedback.cars.get(number)
            if other is None:
                raise RuntimeError('car%d missing; cannot guarantee clearance' % number)
            dx, dy = pose.x - other.x, pose.y - other.y
            vx, vy = self.velocity
            speed2 = vx * vx + vy * vy
            t = max(0., min(.5, -(dx * vx + dy * vy) / speed2)) if speed2 else 0.
            if math.hypot(dx + vx * t, dy + vy * t) < .22:
                raise RuntimeError('predicted clearance from car%d below 22 cm' % number)
        return pose


def move(car, gps, target):
    class Sink:
        def writerow(self, row):
            pass
    from queue import SimpleQueue
    monitor = FleetMonitor(car, gps, Sink(), SimpleQueue(), car.read_status().packet_count,
                           center=(0., -.4, 90.))
    # Transit is field-bounded; calibration's local box is unsuitable for returning south.
    original_radius = calibration.LOCAL_Y_RADIUS
    original_inner_x = calibration.INNER_X
    original_inner_y = calibration.INNER_Y
    calibration.LOCAL_Y_RADIUS = 2.0
    calibration.INNER_X = .72
    # Cars may be staged just beyond the normal 1.30 m playing envelope.
    # Permit only the guarded transit to bring them inward from that row;
    # calibration pulses still use the stricter default boundary.
    calibration.INNER_Y = 1.50
    try:
        monitor.check()
        print('Moving car%d to %s' % (car.number, target), flush=True)
        calibration.drive_to(monitor, target, time.monotonic() + 25., timeout=25.)
    finally:
        car.stop()
        monitor.close()
        calibration.LOCAL_Y_RADIUS = original_radius
        calibration.INNER_X = original_inner_x
        calibration.INNER_Y = original_inner_y


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cars', type=int, nargs='+', default=[2, 3, 4, 1])
    args = parser.parse_args(argv)
    if any(number not in range(1, 5) for number in args.cars):
        parser.error('cars must be IDs 1 through 4')
    calibration.Monitor = FleetMonitor
    root = Path('tools/runs') / time.strftime('%Y%m%dT%H%M%SZ_fleet', time.gmtime())
    root.mkdir(parents=True)
    # The vision server commonly reports 100-220 ms of source age; allow
    # delivery jitter while the onboard freshness flag and clearance guard
    # still fail closed on a real outage.
    gps = GPS(stale_after=.8)
    results = {}
    try:
        time.sleep(1)
        for number in args.cars:
            print('Connecting car%d' % number, flush=True)
            try:
                active = Car(number, timeout=20)
            except (ConnectionError, TimeoutError) as error:
                results[number] = dict(applied=False, error=str(error))
                print('car%d unavailable: %s' % (number, error), flush=True)
                continue
            with active as car:
                car.stop()
                origin = car.wait_for_position(8)
                if len(gps.latest.cars) < 4:
                    raise RuntimeError('all four cars must be visible')
                move(car, gps, (origin.x, 0., 90.))
                move(car, gps, (0., 0., 90.))
                directory = root / ('car%d' % number)
                directory.mkdir()
                try:
                    result = calibration.calibrate(car, gps, directory, auto_apply=True,
                                                   strafe_seconds=1.2, yaw_seconds=1.2)
                    results[number] = dict(applied=True, matrix=result['matrix'])
                except calibration.CalibrationQualityError as error:
                    results[number] = dict(applied=False, error=str(error))
                    print('car%d rejected: %s' % (number, error), flush=True)
                except (RuntimeError, TimeoutError) as error:
                    # A failed sample/recenter is local to this car. Calibration
                    # already stops in its finally block; record the rejection,
                    # return it to staging under the fleet guard, and continue.
                    results[number] = dict(applied=False, error=str(error))
                    print('car%d calibration aborted: %s' % (number, error), flush=True)
                inward_x = max(-.55, min(.55, origin.x))
                move(car, gps, (inward_x, 0., 90.))
                move(car, gps, (inward_x, origin.y, origin.angle))
                move(car, gps, (origin.x, origin.y, origin.angle))
                (root / 'summary.json').write_text(json.dumps(results, indent=2))
    finally:
        Car.close_all()
        gps.close()
        print('Fleet results:', root, json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
