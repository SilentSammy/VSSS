"""Start a shared VSSS provider that replays a recorded bag."""
import argparse
from pathlib import Path

from vsss._bag_provider import BagProvider


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--loop', action='store_true')
    args = parser.parse_args(argv)
    try:
        provider = BagProvider(args.bag, args.speed, args.loop)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f'Bag provider: {args.bag}. Waiting for an observer; run vsss-plot.', flush=True)
    try:
        provider.run()
    except KeyboardInterrupt:
        pass
    except OSError as error:
        parser.exit(1, f'Cannot start bag provider: {error}. '
                       'Another VSSS provider may already be active; close its clients first.\n')


if __name__ == '__main__':
    main()
