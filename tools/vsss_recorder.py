"""Record original UDP datagrams and local BLE operations from VSSS Provider."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

from vsss._provider_observer import ProviderObserver


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='new JSON Lines recording file')
    parser.add_argument('--server', default='192.168.12.1', help='vision server (default: 192.168.12.1)')
    parser.add_argument('--broadcast', action='store_true', help='observe broadcast instead of subscribing')
    parser.add_argument('--bind', default='0.0.0.0', help='provider UDP bind address')
    parser.add_argument('--port', type=int, default=5000, help='vision data port (default: 5000)')
    parser.add_argument('--listen-port', type=int, help='provider UDP port (default: automatic for unicast)')
    parser.add_argument('--subscription-port', type=int, help='subscription port (default: data port + 1)')
    parser.add_argument('--duration', type=float, help='stop after this many seconds')
    parser.add_argument('--topics', nargs='+', choices=('udp', 'ble'), default=['udp', 'ble'],
                        help='streams to record (default: udp ble)')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or (args.listen_port is not None and not 0 <= args.listen_port <= 65535):
        parser.error('invalid UDP port')
    if args.subscription_port is not None and not 1 <= args.subscription_port <= 65535:
        parser.error('invalid subscription port')
    if not args.broadcast and args.subscription_port is None and args.port == 65535:
        parser.error('subscription port is required when data port is 65535')
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error('duration must be positive and finite')
    vision = dict(server=None if args.broadcast else args.server, port=args.port, bind=args.bind,
                  listen_port=args.listen_port if args.listen_port is not None else (args.port if args.broadcast else 0),
                  subscription_port=args.subscription_port or args.port + 1)
    counts = {}
    with args.output.open('x', encoding='utf-8') as output:
        def save(event):
            output.write(json.dumps(event, separators=(',', ':')) + '\n')
            output.flush()
            counts[event['type']] = counts.get(event['type'], 0) + 1

        complete = False
        save(dict(type='session_start', schema=2, topics=args.topics,
                  monotonic_ns=time.monotonic_ns(), wall_time_ns=time.time_ns()))
        try:
            with ProviderObserver(args.topics, vision) as observer:
                save(observer.snapshot)
                print('Recording to', args.output, flush=True)
                deadline = time.monotonic() + args.duration if args.duration else None
                while deadline is None or time.monotonic() < deadline:
                    event = observer.read(.1)
                    if event is None:
                        continue
                    save(event)
                    if event['type'] == 'observer_overflow':
                        raise RuntimeError('Provider event queue overflow; recording is incomplete')
            complete = True
        except KeyboardInterrupt:
            complete = True
        finally:
            save(dict(type='session_end', complete=complete, counts=dict(counts),
                      monotonic_ns=time.monotonic_ns(), wall_time_ns=time.time_ns()))
    print('Recorded', sum(counts.values()), 'events.')
    if 'udp' in args.topics and not counts.get('udp'):
        print('Warning: no UDP telemetry was received.', file=sys.stderr)


if __name__ == '__main__':
    main()
