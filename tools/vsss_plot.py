"""Plot live or recorded VSSS field telemetry using the aruco_gps renderer."""
import argparse
import math
from pathlib import Path
import time

from vsss.provider import ProviderClient


def provider_from_args(args):
    vision = dict(server=None if args.broadcast else args.server, port=args.port,
                  listen_port=args.port if args.broadcast else 0,
                  subscription_port=args.subscription_port or args.port + 1)
    if args.metadata_url:
        vision['metadata_url'] = args.metadata_url
    return ProviderClient(('udp',), vision, auto_start=not args.persist,
                          connect_timeout=.1 if args.persist else 8)


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
    parser.add_argument('--broadcast', action='store_true')
    parser.add_argument('--server', default='192.168.12.1')
    parser.add_argument('--metadata-url', help='override the vision HTTP status endpoint')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--subscription-port', type=int)
    parser.add_argument('--field-width', type=float, help='fallback width when metadata is unavailable')
    parser.add_argument('--field-height', type=float, help='fallback height when metadata is unavailable')
    parser.add_argument('--width', type=int, default=720)
    parser.add_argument('--duration', type=float)
    parser.add_argument('--persist', action='store_true',
                        help='keep running across feed endings and provider restarts; '
                             'wait for providers without automatically starting live mode')
    parser.add_argument('--output', type=Path, help='headless: save the final frame as a new image')
    args = parser.parse_args(argv)
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
    provider = None
    retry_at = 0.
    try:
        if args.output is None:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while deadline is None or time.monotonic() < deadline:
            try:
                if provider is None and time.monotonic() >= retry_at:
                    provider = provider_from_args(args)
                    plotter = plotter_from_snapshot(provider.snapshot, args.width, fallback)
                    latest = None
                event = provider.read(.02) if provider is not None else None
                if event and event.get('type') == 'stream_end' and not event.get('complete', True):
                    raise RuntimeError(event.get('error', 'Provider stream ended incomplete'))
                if event and event.get('type') == 'udp' and event.get('valid'):
                    latest = plotter.render_packet(event['packet'])
                if provider is not None and provider.finished:
                    if not args.persist:
                        break
                    provider.close()
                    provider = None
                    retry_at = time.monotonic() + .5
            except (ConnectionError, OSError):
                if not args.persist:
                    raise
                if provider is not None:
                    provider.close()
                    provider = None
                retry_at = time.monotonic() + .5
            if args.output is None:
                if latest is not None:
                    cv2.imshow(window, latest)
                if cv2.waitKey(1) & 0xff in (ord('q'), 27):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if provider is None:
                time.sleep(.02)
    except KeyboardInterrupt:
        pass
    finally:
        if provider is not None:
            provider.close()
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
