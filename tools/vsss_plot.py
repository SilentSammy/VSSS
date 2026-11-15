"""Plot live or recorded VSSS field telemetry using the aruco_gps renderer."""
import argparse
import math
from pathlib import Path
import time

from vsss.provider import BagProvider, LiveProvider


def provider_from_args(args):
    if args.bag:
        return BagProvider(args.bag, topics=('udp',), speed=args.speed)
    vision = dict(server=None if args.broadcast else args.server, port=args.port,
                  listen_port=args.port if args.broadcast else 0,
                  subscription_port=args.subscription_port or args.port + 1)
    if args.metadata_url:
        vision['metadata_url'] = args.metadata_url
    return LiveProvider(('udp',), vision)


def plotter_from_snapshot(snapshot, width, fallback=None):
    from aruco_gps.plotter2d import BoardPlotter2D
    vision = snapshot.get('vision') or {}
    status = (vision.get('metadata') or {}).get('status') or {}
    setup = status.get('setup')
    if not setup:
        if fallback is None:
            raise ValueError('No vision setup metadata. Supply --field-width and --field-height '
                             'for an older bag or an unavailable status endpoint.')
        setup = dict(board=dict(dimensions_m=list(fallback), print_dimensions_m=list(fallback)),
                     marker_size_m=.09, plot_margin_m=.05, ball_detector=dict(radius_m=.02))
    return BoardPlotter2D.from_metadata(setup, width_px=width)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--bag', type=Path, help='replay a vsss-observer JSONL bag')
    source.add_argument('--broadcast', action='store_true')
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--server', default='192.168.12.1')
    parser.add_argument('--metadata-url', help='override the vision HTTP status endpoint')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--subscription-port', type=int)
    parser.add_argument('--field-width', type=float, help='fallback width when metadata is unavailable')
    parser.add_argument('--field-height', type=float, help='fallback height when metadata is unavailable')
    parser.add_argument('--width', type=int, default=720)
    parser.add_argument('--duration', type=float)
    parser.add_argument('--output', type=Path, help='headless: save the final frame as a new image')
    args = parser.parse_args(argv)
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed must be positive and finite')
    if not 1 <= args.port <= 65534:
        parser.error('--port must be 1..65534')
    if args.subscription_port is not None and not 1 <= args.subscription_port <= 65535:
        parser.error('invalid subscription port')
    if args.width < 100:
        parser.error('--width must be at least 100')
    if (args.field_width is None) != (args.field_height is None):
        parser.error('supply both fallback field dimensions')
    fallback = (args.field_width, args.field_height) if args.field_width is not None else None
    if fallback and not all(math.isfinite(value) and value > 0 for value in fallback):
        parser.error('field dimensions must be positive and finite')
    if args.output and args.output.exists():
        parser.error('--output must be a new file')
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error('--duration must be positive and finite')
    import cv2
    latest = None
    window = 'VSSS field'
    started = time.monotonic()
    deadline = started + args.duration if args.duration else None
    try:
        with provider_from_args(args) as provider:
            plotter = plotter_from_snapshot(provider.snapshot, args.width, fallback)
            if args.output is None:
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            while not provider.finished and (deadline is None or time.monotonic() < deadline):
                event = provider.read(.02)
                if event and event.get('type') == 'udp' and event.get('valid'):
                    latest = plotter.render_packet(event['packet'])
                if args.output is None:
                    if latest is not None:
                        cv2.imshow(window, latest)
                    if cv2.waitKey(1) & 0xff in (ord('q'), 27):
                        break
                    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        break
    except KeyboardInterrupt:
        pass
    finally:
        if args.output is None:
            cv2.destroyAllWindows()
    if args.output:
        if latest is None:
            raise RuntimeError('provider produced no valid UDP frames')
        if not cv2.imwrite(str(args.output), latest):
            raise OSError(f'could not write plot image: {args.output}')
        print('Saved', args.output)


if __name__ == '__main__':
    main()
