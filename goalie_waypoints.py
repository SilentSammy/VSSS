"""Experimental goalie: repeatedly target the ball's current X with a waypoint.

Run against a simulator first: python goalie_waypoints.py
No prediction; the onboard controller handles heading and lane corrections.
"""
import argparse
import math
from time import monotonic, sleep

from vsss import Car, run


def defend(car, *, y=-1.2, heading=90, interval=.05, duration=0):
    deadline = monotonic() + duration if duration else math.inf
    try:
        car.wait_for_position()
        while monotonic() < deadline:
            balls = car.gps.balls
            if balls and car.pose is not None:
                ball = balls[0]
                car.go_to(x=max(-.65, min(.65, ball.x)), y=y, angle=heading)
            else:
                car.stop()
            sleep(interval)
    finally:
        car.stop()


@run
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--car', type=int, default=1)
    parser.add_argument('--y', type=float, default=-1.2, help='Fixed field Y in metres')
    parser.add_argument('--heading', type=float, default=90, help='Fixed heading in degrees')
    parser.add_argument('--interval', type=float, default=.05, help='Seconds between waypoints')
    parser.add_argument('--duration', type=float, default=0, help='Run seconds; 0 runs until Ctrl+C')
    parser.add_argument('--endpoint', help='Optional provider discovery file')
    args = parser.parse_args(argv)
    if (args.car < 0 or not all(math.isfinite(v) for v in (args.y,args.heading,args.interval,args.duration))
            or args.interval <= 0 or args.duration < 0):
        parser.error('Use finite values, a positive interval, and nonnegative car/duration.')
    print(f'Following ball X at Y={args.y:g}, heading={args.heading:g}, every {args.interval:g}s. Ctrl+C stops.')
    defend(Car(args.car, endpoint=args.endpoint), y=args.y, heading=args.heading,
           interval=args.interval, duration=args.duration)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nGoalie stopped.')
