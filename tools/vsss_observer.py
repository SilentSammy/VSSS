"""Observe UDP/BLE traffic through VSSS Provider, optionally recording JSON Lines."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import struct
from pathlib import Path
import sys
import time

from vsss._provider_observer import ProviderObserver
from vsss import _ble_protocol as protocol


def format_packet(message, sender):
    """Readable world state; the provider has already validated its fields."""
    age = message.get('frame_age_ms')
    timing = f' age={float(age):.1f}ms' if age is not None else ''
    positions = [f"ID {int(p['id'])}: x={float(p['x']):+.3f}m, y={float(p['y']):+.3f}m, "
                 f"angle={float(p['angle']):+.1f}deg" for p in message['markers']]
    positions += [f"ball: x={float(p['x']):+.3f}m, y={float(p['y']):+.3f}m"
                  for p in message.get('balls', [])]
    return f"{sender[0]} board={'yes' if message['board_detected'] else 'no'}{timing} | " + (' | '.join(positions) or 'no objects')


def _ble_value(event):
    if 'value' in event:
        return event['value']
    payload = event.get('payload_hex')
    if payload is None:
        return None
    try:
        data = bytes.fromhex(payload)
        characteristic = event.get('characteristic')
        if characteristic == 'status':
            status = protocol.RobotStatus(*protocol.STATUS_STRUCT.unpack(data))
            return dict(asdict(status), mode=status.mode_name, fault=status.fault_name)
        if characteristic == 'trim':
            return protocol.TRIM_STRUCT.unpack(data)
        if characteristic == 'controller':
            return dict(zip(protocol.CONTROLLER_KEYS, protocol.CONTROLLER_STRUCT.unpack(data)))
    except (ValueError, struct.error):
        pass  # Future/unknown payload formats remain visible as raw hex.
    return payload


def format_event(event):
    kind = event['type']
    if kind == 'udp':
        if not event['valid']:
            return f"UDP {event['sender'][0]} invalid: {event['error']}"
        return 'UDP ' + format_packet(event['packet'], event['sender'])
    if kind == 'ble':
        detail = _ble_value(event)
        reason = f" ({event['reason']})" if event.get('reason') else ''
        result = f" {json.dumps(detail)}" if detail is not None else ''
        if event.get('error'):
            result = ' ' + event['error']
        return (f"BLE {event['car']} {event['operation']} {event.get('characteristic', event.get('uuid', '?'))} "
                f"#{event.get('request', '?')} {event['phase']}{reason}{result}")
    if kind == 'provider_snapshot':
        vision = event.get('vision')
        source = f"; UDP port {vision['local_port']}" if vision else ''
        return 'Provider: controlled cars=' + ', '.join(event['controlled_cars']) + source
    if kind == 'bag_header':
        return format_event(event['provider'])
    details = {k: v for k, v in event.items() if k not in ('type', 'monotonic_ns', 'wall_time_ns')}
    return kind + ': ' + json.dumps(details)


def main(argv=None, *, stop_event=None, ready_event=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='also record full events to a new JSON Lines file')
    parser.add_argument('--quiet', action='store_true', help='suppress individual events (requires --output)')
    parser.add_argument('--server', default='192.168.12.1', help='vision server (default: 192.168.12.1)')
    parser.add_argument('--metadata-url', help='override the vision HTTP status endpoint')
    parser.add_argument('--broadcast', action='store_true', help='observe broadcast instead of subscribing')
    parser.add_argument('--bind', default='0.0.0.0', help='provider UDP bind address')
    parser.add_argument('--port', type=int, default=5000, help='vision data port (default: 5000)')
    parser.add_argument('--listen-port', type=int, help='provider UDP port (default: automatic for unicast)')
    parser.add_argument('--subscription-port', type=int, help='subscription port (default: data port + 1)')
    parser.add_argument('--duration', type=float, help='stop after this many seconds')
    parser.add_argument('--topics', nargs='+', choices=('udp', 'ble'), default=['udp', 'ble'],
                        help='streams to observe (default: udp ble)')
    args = parser.parse_args(argv)
    if args.quiet and args.output is None:
        parser.error('--quiet requires --output')
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
    if args.metadata_url:
        vision['metadata_url'] = args.metadata_url
    counts = {}
    with args.output.open('x', encoding='utf-8') if args.output else nullcontext(None) as output:
        def save(event):
            if output is not None:
                output.write(json.dumps(event, separators=(',', ':')) + '\n')
                output.flush()
            counts[event['type']] = counts.get(event['type'], 0) + 1
            if not args.quiet and event['type'] not in ('session_start', 'session_end'):
                print(format_event(event), flush=True)

        complete = False
        try:
            with ProviderObserver(args.topics, vision) as observer:
                save(dict(type='bag_header', schema=3, topics=args.topics,
                          provider=observer.snapshot,
                          monotonic_ns=time.monotonic_ns(), wall_time_ns=time.time_ns()))
                if args.output:
                    print('Recording to', args.output, flush=True)
                if ready_event is not None:
                    ready_event.set()
                deadline = time.monotonic() + args.duration if args.duration else None
                while deadline is None or time.monotonic() < deadline:
                    if stop_event is not None and stop_event.is_set():
                        break
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
    print('Observed', counts.get('udp', 0), 'UDP and', counts.get('ble', 0), 'BLE events.')
    if 'udp' in args.topics and not counts.get('udp'):
        print('Warning: no UDP telemetry was received.', file=sys.stderr)


if __name__ == '__main__':
    main()
