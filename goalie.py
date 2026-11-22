"""Feedback goalie: defend a fixed Y lane, with small lane/heading corrections.

Try: vsss-sim --scenario goalie, then python goalie.py.
Assumes one incoming ball, a straight shot without wall bounces, and a clear lane.
"""
import argparse
import math
from time import monotonic, sleep

from vsss import Car, gps, run


def intercept(previous, current, line_y, limit=.63):
    """Two (time, x, y) samples predict X at the goalie's fixed Y."""
    dt = current[0] - previous[0]
    dx, dy = current[1] - previous[1], current[2] - previous[2]
    if dt <= 0 or dy / dt <= .05 or current[2] >= line_y:
        return None
    x = current[1] + dx / dy * (line_y - current[2])
    return max(-limit, min(limit, x))


def defend(car, duration=0., width=1.5):
    """Predict X crossing and hold the original lane/heading under contacts."""
    try:
        pose = car.wait_for_position()
        if abs((pose.angle + 180) % 360 - 180) > 3:
            raise RuntimeError('Start the goalie facing +X (angle 0), so forward motion changes only X.')
        line_y, limit = pose.y, width / 2 - .12
        previous = None
        target = pose.x
        deadline = monotonic() + duration if duration else math.inf
        # This explicit command releases scenarios waiting for student control.
        car.set_velocity(x=0., y=0., w=0.)
        while monotonic() < deadline:
            pose, frame = car.pose, gps.latest
            if pose is None or frame is None or gps.feedback_problem or not frame.board_detected or len(frame.balls) != 1:
                raise RuntimeError('Fresh car/board feedback and one visible ball are required.')
            heading = (pose.angle + 180) % 360 - 180
            if abs(pose.y - line_y) > .05 or abs(heading) > 15:
                raise RuntimeError('The goalie left its lane/heading recovery band; reset its pose.')
            ball = frame.balls[0]
            now = frame.server_timestamp if frame.server_timestamp is not None else frame.received_at
            current = (now, ball.x, ball.y)
            if previous is not None and now - previous[0] >= .08:
                target = intercept(previous, current, line_y, limit)
                previous = current
            elif previous is None:
                previous = current
            command = 0. if target is None else max(-1., min(1., 5 * (target - pose.x)))
            # Dynamic contacts can perturb the old kinematic example's assumed
            # heading. Hold the lane in field coordinates and transform back
            # into the car frame; do not tune the physics to hide the disturbance.
            lateral = max(-.3, min(.3, 4 * (line_y-pose.y)))
            theta = math.radians(heading)
            x = command*math.cos(theta) + lateral*math.sin(theta)
            y = -command*math.sin(theta) + lateral*math.cos(theta)
            car.set_velocity(x=max(-1.,min(1.,x)), y=max(-1.,min(1.,y)),
                             w=max(-.35,min(.35,-.03*heading)))
            sleep(.04)
    finally:
        car.stop()


@run
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--car', type=int, default=1)
    parser.add_argument('--duration', type=float, default=0., help='seconds to defend; 0 runs until Ctrl+C')
    parser.add_argument('--width', type=float, default=1.5, help='field width in metres')
    args = parser.parse_args(argv)
    if args.car < 0 or not math.isfinite(args.duration) or args.duration < 0 or not math.isfinite(args.width) or args.width <= .3:
        parser.error('use a nonnegative car ID/duration and a positive finite field width')
    defend(Car(args.car), args.duration, args.width)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nGoalie stopped.')
