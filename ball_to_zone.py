"""A student example: get behind one ball, then push steadily into a goal zone.

Uses only Car, gps and run. Running this script moves the selected car.
Start vsss-sim first to practice. Coordinates are metres; headings are degrees.
"""
import argparse
import math
from time import monotonic, sleep

from vsss import Car, gps, run


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def observe(car):
    """Stop rather than drive blind or guess which ball to follow."""
    pose, balls = car.pose, gps.balls
    if pose is None or gps.feedback_problem or not gps.board_detected:
        raise RuntimeError('Fresh field feedback is required: ' +
                           (gps.feedback_problem or 'car marker or board is not visible.'))
    if len(balls) != 1:
        raise RuntimeError('This example requires exactly one visible ball.')
    return pose, (balls[0].x, balls[0].y)


def check_field(pose, ball, field):
    """Do not keep feeding motion or plan a chase outside the usable field."""
    if any(abs(v) > size / 2 - .1 for v, size in zip((pose.x, pose.y), field)):
        raise RuntimeError('Car is too close to the field boundary.')
    if any(abs(v) > size / 2 - .025 for v, size in zip(ball, field)):
        raise RuntimeError('Ball is at or outside the field boundary; check its position/detection before continuing.')


def approach_points(position, ball, direction, clearance=.23):
    """Walk around the ball on a small circle when it blocks the approach."""
    behind = (ball[0] - clearance * direction[0], ball[1] - clearance * direction[1])
    segment = behind[0] - position[0], behind[1] - position[1]
    length2 = segment[0] ** 2 + segment[1] ** 2
    fraction = max(0., min(1., ((ball[0] - position[0]) * segment[0] +
                              (ball[1] - position[1]) * segment[1]) / max(length2, 1e-9)))
    nearest = position[0] + fraction * segment[0], position[1] + fraction * segment[1]
    if distance(nearest, ball) >= .16:
        return [behind]
    start = math.atan2(position[1] - ball[1], position[0] - ball[0])
    end = math.atan2(-direction[1], -direction[0])
    turn = (end - start + math.pi) % (2 * math.pi) - math.pi
    count = max(1, math.ceil(abs(turn) / (math.pi / 3)))
    # First move radially outward, then follow short chords around the ball.
    angles = [start] + [start + turn * i / count for i in range(1, count + 1)]
    return [(ball[0] + clearance * math.cos(a), ball[1] + clearance * math.sin(a)) for a in angles]


def go_to(car, point, heading, deadline, field=(1.5, 2.9)):
    car.go_to(point[0], point[1], heading)
    while monotonic() < deadline:
        check_field(*observe(car), field)
        status = car.read_status()
        if status.fault:
            raise RuntimeError(f'Car controller fault: {status.fault_name}')
        if not status.pose_fresh:
            raise RuntimeError('Onboard camera pose is stale, even if host GPS is fresh.')
        if status.mode_name == 'reached':
            return
        sleep(.08)
    raise TimeoutError('Approach timed out.')


def push_velocity(pose, ball, target, push_speed=.4, field=(1.5, 2.9), contact=False):
    """Continuous moderate feed; avoid delayed stop/sideways/turn oscillations."""
    remaining = distance(ball, target)
    if remaining < 1e-9:
        return dict(x=0., y=0., w=0.)
    ux, uy = (target[0] - ball[0]) / remaining, (target[1] - ball[1]) / remaining
    dx, dy = ball[0] - pose.x, ball[1] - pose.y
    theta = math.radians(pose.angle)
    heading_error = (math.atan2(uy, ux) - theta + math.pi) % (2 * math.pi) - math.pi
    side_error = -uy * dx + ux * dy
    body_forward = math.cos(theta) * dx + math.sin(theta) * dy
    body_side = -math.sin(theta) * dx + math.cos(theta) * dy
    if (math.hypot(dx, dy) > .4 or dx * ux + dy * uy < -.03 or
            abs(side_error) > .09 or abs(heading_error) > math.radians(30) or
            (body_forward < .13 and abs(body_side) > .06)):
        return None
    facing = math.cos(theta), math.sin(theta)
    if any(abs(v) > size / 2 - .18 and v * direction > .02
           for v, direction, size in zip(ball, facing, field)):
        return None  # A correction must not keep pushing the ball toward a nearby edge.
    # Bag 12 spent most of its short strokes at the former .20/.25 floor.
    # Keep useful forward motion instead of toggling a centering mode on delayed poses.
    forward = min(push_speed, max(.3, .5 * remaining))
    if body_forward > .15:
        forward = min(forward, .3)
    lateral = max(-.1, min(.1, 3 * side_error))
    vx, vy = forward * ux - lateral * uy, forward * uy + lateral * ux
    # Turning a square bumper against the ball adds a tangential kick. Once
    # contact is seen, move_ball latches this until the next waypoint approach.
    touching = contact or math.hypot(dx, dy) < .18
    return dict(x=max(0., min(forward, math.cos(theta) * vx + math.sin(theta) * vy)),
                y=max(-.1, min(.1, -math.sin(theta) * vx + math.cos(theta) * vy)),
                # The public API rounds to .05: no minimum yaw step at a tight
                # threshold. Lower gain/limit reduces delayed over-correction.
                w=0. if touching else max(-.05, min(.05, .2 * heading_error)))


def wait_for_still_ball(car, deadline, field=(1.5, 2.9)):
    """Plan the next approach after a loose ball stops, not at its old position."""
    pose, anchor = observe(car)
    check_field(pose, anchor, field)
    stable_since = monotonic()
    while monotonic() < deadline:
        sleep(.1)
        pose, ball = observe(car)
        check_field(pose, ball, field)
        if distance(ball, anchor) > .012:
            anchor, stable_since = ball, monotonic()
        if monotonic() - stable_since >= .6:
            return
    raise TimeoutError('Ball did not settle before replanning.')


def move_ball(car, target=(0., 1.), radius=.18, timeout=60., field=(1.5, 2.9), push_speed=.4):
    """Return the final observed ball position after it settles inside the zone.

    This small demonstration assumes one ball, a clear field, a 14 cm car and
    a target with room to stand behind the ball. It is not a match planner.
    """
    deadline = monotonic() + timeout
    try:
        car.wait_for_position(timeout=min(10., timeout))
        while monotonic() < deadline:
            pose, ball = observe(car)
            check_field(pose, ball, field)
            if distance(ball, target) < radius:
                car.stop()
                # Check the ball after its final coast, not just as it crosses the zone.
                settled = monotonic()
                anchor = ball
                while monotonic() < deadline:
                    sleep(.1)
                    pose, ball = observe(car)
                    check_field(pose, ball, field)
                    if distance(ball, target) > radius:
                        wait_for_still_ball(car, deadline, field)
                        break
                    if distance(ball, anchor) > .012:
                        settled = monotonic()
                        anchor = ball
                    if monotonic() - settled >= 1.2:
                        return ball
                continue
            length = distance(ball, target)
            direction = (target[0] - ball[0]) / length, (target[1] - ball[1]) / length
            heading = math.degrees(math.atan2(direction[1], direction[0]))
            points = approach_points((pose.x, pose.y), ball, direction)
            if any(abs(x) > field[0] / 2 - .11 or abs(y) > field[1] / 2 - .11 for x, y in points):
                raise RuntimeError('Not enough room behind/around the ball for this simple approach.')
            print('Get behind the ball.', flush=True)
            for point in points:
                go_to(car, point, heading, deadline, field)
            print('Push steadily toward the zone.', flush=True)
            contact = False
            while monotonic() < deadline:
                pose, ball = observe(car)
                check_field(pose, ball, field)
                remaining = distance(ball, target)
                # Firm contact carries more momentum: release near the zone's
                # entrance and let the ball coast toward its center.
                if remaining < radius * .95:
                    break
                contact = contact or distance((pose.x, pose.y), ball) < .18
                command = push_velocity(pose, ball, target, push_speed, field, contact=contact)
                if command is None:
                    print(f'Contact/alignment lost at ball ({ball[0]:+.3f}, {ball[1]:+.3f}); '
                          'stop and let it settle before re-approaching.', flush=True)
                    car.stop()
                    wait_for_still_ball(car, deadline, field)
                    break
                car.set_velocity(**command)
                sleep(.08)
        raise TimeoutError('Ball did not settle in the zone before the time limit.')
    finally:
        car.stop()


@run
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--car', type=int, default=1)
    parser.add_argument('--target', type=float, nargs=2, default=(0., 1.), metavar=('X', 'Y'))
    parser.add_argument('--radius', type=float, default=.18, help='goal zone radius in metres')
    parser.add_argument('--timeout', type=float, default=60.)
    parser.add_argument('--push-speed', type=float, default=.4, help='normalized cruise command, 0.25..0.6 (default: 0.4)')
    parser.add_argument('--field', type=float, nargs=2, default=(1.5, 2.9), metavar=('WIDTH', 'HEIGHT'))
    args = parser.parse_args(argv)
    if args.car < 0 or not all(math.isfinite(v) for v in (*args.target, args.radius, args.timeout, args.push_speed, *args.field)):
        parser.error('use finite coordinates and a nonnegative car ID')
    if args.radius < .08 or args.timeout <= 0 or min(args.field) <= .3:
        parser.error('radius must be at least .08 m; timeout and field dimensions must be positive')
    if any(abs(v) + args.radius > size / 2 for v, size in zip(args.target, args.field)):
        parser.error('goal zone must fit inside the field')
    if not .25 <= args.push_speed <= .6:
        parser.error('--push-speed must be in 0.25..0.6')
    car = Car(args.car)
    result = move_ball(car, tuple(args.target), args.radius, args.timeout, tuple(args.field), args.push_speed)
    print(f'Ball settled in the zone: ({result[0]:+.3f}, {result[1]:+.3f}) m.', flush=True)


if __name__ == '__main__':
    main()
