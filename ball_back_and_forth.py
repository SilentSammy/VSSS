"""Move one ball between (0, +1) and (0, -1), repeating until Ctrl+C.

Start vsss-sim first to practice. Uses the public-API controller in ball_to_zone.
Running this demo moves the selected car; use a clear field and one ball.
"""
import argparse
import math
from time import sleep

from vsss import Car, run
from ball_to_zone import move_ball


@run
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--car', type=int, default=1)
    parser.add_argument('--cycles', type=int, default=0,
                        help='round trips to perform; 0 repeats until Ctrl+C (default)')
    parser.add_argument('--push-speed', type=float, default=.4, help='normalized cruise command, 0.25..0.6')
    parser.add_argument('--field', type=float, nargs=2, default=(1.5, 2.9), metavar=('WIDTH', 'HEIGHT'))
    args = parser.parse_args(argv)
    if args.car < 0 or args.cycles < 0:
        parser.error('car ID and cycle count must be nonnegative')
    if not math.isfinite(args.push_speed) or not .25 <= args.push_speed <= .6:
        parser.error('--push-speed must be in 0.25..0.6')
    if not all(math.isfinite(v) and v > .3 for v in args.field) or any(
            abs(v) + .18 > size / 2 for v, size in zip((0., 1.), args.field)):
        parser.error('field must contain both 18 cm target zones at y=+/-1')

    car = Car(args.car)
    cycle = 0
    while args.cycles == 0 or cycle < args.cycles:
        for y in (1.0, -1.0):
            print(f'Round trip {cycle + 1}: moving ball to y={y:+.1f} m.', flush=True)
            x, actual_y = move_ball(car, target=(0., y), radius=.18, timeout=90.,
                                   push_speed=args.push_speed, field=tuple(args.field))
            print(f'Ball settled at ({x:+.3f}, {actual_y:+.3f}) m. Pausing.', flush=True)
            sleep(1)
        cycle += 1


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nDemo stopped.')
