"""Happy-path kick: the straight path behind the stationary ball must be clear.

Uses the same CLI and helpers as kick_ball.py; intentionally skips detour planning.
"""
from kick_ball import aim, line_up, main, strike


def kick(car, target, field=(1.5, 2.9)):
    try:
        car.wait_for_position()
        _, ball, direction, heading = aim(car, target, field)
        behind = tuple(ball[i] - .30 * direction[i] for i in (0, 1))
        line_up(car, [behind], ball, heading, field)
        strike(car, ball, direction, field)
    finally:
        car.stop()


if __name__ == '__main__':
    main(kick)
