"""One full-power kick toward a point; no attempt to stop the ball there.

Practice with vsss-sim first. Uses the student API and shared observation /
waypoint helpers from ball_to_zone. Assumes one stationary ball and a clear field.
"""
import argparse
import math
from time import monotonic, sleep

from vsss import Car, gps, run
from ball_to_zone import check_field, distance, go_to, observe


def aim(car, target, field):
    pose, ball = observe(car)
    check_field(pose, ball, field)
    length = distance(ball, target)
    if length < .05:
        raise ValueError('Aim at a point at least 5 cm from the ball.')
    direction = ((target[0] - ball[0]) / length, (target[1] - ball[1]) / length)
    heading = math.degrees(math.atan2(direction[1], direction[0]))
    return pose, ball, direction, heading


def approach(position, ball, direction):
    """A clear straight approach, or three sides of a rectangle around the ball."""
    ux, uy = direction
    nx, ny = -uy, ux
    dx, dy = position[0] - ball[0], position[1] - ball[1]
    along, side = dx * ux + dy * uy, dx * nx + dy * ny
    behind = (ball[0] - .30 * ux, ball[1] - .30 * uy)
    # A segment stays clear if its closest point is outside this small disk.
    sx, sy = behind[0] - position[0], behind[1] - position[1]
    fraction = max(0., min(1., -(dx * sx + dy * sy) / max(sx * sx + sy * sy, 1e-9)))
    if math.hypot(dx + fraction * sx, dy + fraction * sy) >= .18:
        return [behind]
    offset = math.copysign(.30, side)
    return [(ball[0] + along * ux + offset * nx, ball[1] + along * uy + offset * ny),
            (behind[0] + offset * nx, behind[1] + offset * ny), behind]


def line_up(car, points, ball, heading, field):
    # Validate the whole route before moving. No general obstacle/edge planner.
    if any(abs(v) > size / 2 - .12 for point in points for v, size in zip(point, field)):
        raise RuntimeError('Not enough field space for this approach.')
    deadline = monotonic() + 60
    for point in points:
        go_to(car, point, heading, deadline, field)
    pose, latest = observe(car)
    if distance(ball, latest) > .025:
        raise RuntimeError('The ball moved during the approach; aim again.')
    if abs((pose.angle - heading + 180) % 360 - 180) > 8:
        raise RuntimeError('The car is not aligned with the kick direction.')


def strike(car, ball, direction, field):
    """Short straight run-through, full command, no turning or ball pursuit."""
    finish = tuple(ball[i] + .08 * direction[i] for i in (0, 1))
    if any(abs(v) > size / 2 - .12 for v, size in zip(finish, field)):
        raise RuntimeError('Not enough field space for the follow-through.')
    deadline = monotonic() + 1.5
    try:
        while monotonic() < deadline:
            pose = car.pose
            if pose is None or gps.feedback_problem or not gps.board_detected:
                raise RuntimeError('Fresh car/board feedback is required during the kick.')
            check_field(pose, ball, field)
            along = (pose.x - ball[0]) * direction[0] + (pose.y - ball[1]) * direction[1]
            if along >= .02:
                return
            car.set_velocity(x=1., y=0., w=0.)
            sleep(.04)
        raise TimeoutError('Kick run-up timed out.')
    finally:
        car.stop()


def kick(car, target, field=(1.5, 2.9)):
    """Line up once, kick once, finish. The target is a direction, not a destination."""
    try:
        car.wait_for_position()
        pose, ball, direction, heading = aim(car, target, field)
        points = approach((pose.x, pose.y), ball, direction)
        line_up(car, points, ball, heading, field)
        strike(car, ball, direction, field)
    finally:
        car.stop()


@run
def main(action=kick, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--car', type=int, default=1)
    parser.add_argument('--target', nargs=2, type=float, default=(0., 1.), metavar=('X', 'Y'))
    parser.add_argument('--field', nargs=2, type=float, default=(1.5, 2.9), metavar=('WIDTH', 'HEIGHT'))
    args = parser.parse_args(argv)
    if args.car < 0 or not all(math.isfinite(v) for v in (*args.target, *args.field)) or min(args.field) <= .6:
        parser.error('use finite coordinates, positive field dimensions and a nonnegative car ID')
    car = Car(args.car)
    action(car, tuple(args.target), tuple(args.field))
    print('Kick complete. The ball may roll past the aim point.', flush=True)


if __name__ == '__main__':
    main()
