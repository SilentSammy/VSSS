"""Record local VSSS BLE operations and vision UDP datagrams as JSON Lines."""
import argparse
import base64
import json
import math
from pathlib import Path
import select
import socket
import sys
import time

from vsss._provider_connection import connect


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='new JSON Lines recording file')
    parser.add_argument('--server', default='192.168.12.1', help='vision server (default: 192.168.12.1)')
    parser.add_argument('--broadcast', action='store_true', help='listen for broadcast instead of subscribing')
    parser.add_argument('--bind', default='0.0.0.0', help='local UDP bind address')
    parser.add_argument('--port', type=int, default=5000, help='vision data port (default: 5000)')
    parser.add_argument('--listen-port', type=int, help='local UDP port (default: automatic for unicast)')
    parser.add_argument('--subscription-port', type=int, help='vision subscription port (default: data port + 1)')
    parser.add_argument('--duration', type=float, help='stop after this many seconds')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or (args.listen_port is not None and not 0 <= args.listen_port <= 65535):
        parser.error('invalid UDP port')
    if args.subscription_port is not None and not 1 <= args.subscription_port <= 65535:
        parser.error('invalid subscription port')
    if not args.broadcast and args.subscription_port is None and args.port == 65535:
        parser.error('subscription port is required when data port is 65535')
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error('duration must be positive and finite')
    subscription = None if args.broadcast else (args.server, args.subscription_port or args.port + 1)
    local_port = args.listen_port if args.listen_port is not None else (args.port if args.broadcast else 0)
    count = 0
    counts = {}
    with args.output.open('x', encoding='utf-8') as output, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
        udp.bind((args.bind, local_port))
        observer = connect('observe')
        observer.settimeout(5)
        hello = b''
        while not hello.endswith(b'\n'):
            part = observer.recv(1)
            if not part or len(hello) > 1024:
                raise ConnectionError('BLE provider did not accept observer')
            hello += part
        if json.loads(hello) != {'provider': 'observing'}:
            raise ConnectionError('BLE provider rejected observer')
        observer.setblocking(False)
        buffer = b''
        next_renewal = 0.
        deadline = time.monotonic() + args.duration if args.duration else None
        def save(event):
            nonlocal count
            output.write(json.dumps(event, separators=(',', ':')) + '\n')
            output.flush()
            count += 1
            counts[event['type']] = counts.get(event['type'], 0) + 1
        save(dict(type='session_start', schema=1, monotonic_ns=time.monotonic_ns(),
                  wall_time_ns=time.time_ns(), udp_bind=list(udp.getsockname()), subscription=subscription))
        print('Recording to', args.output, '; UDP port', udp.getsockname()[1], flush=True)
        complete = False
        try:
            while deadline is None or time.monotonic() < deadline:
                now = time.monotonic()
                if subscription and now >= next_renewal:
                    udp.sendto(b'POSE_SUBSCRIBE', subscription)
                    next_renewal = now + 1.
                ready, _, _ = select.select([udp, observer], [], [], .1)
                if udp in ready:
                    data, sender = udp.recvfrom(65535)
                    event = dict(type='udp', monotonic_ns=time.monotonic_ns(), wall_time_ns=time.time_ns(),
                                 sender=list(sender), payload_base64=base64.b64encode(data).decode('ascii'))
                    try:
                        event['packet'] = json.loads(data)
                    except (UnicodeDecodeError, ValueError):
                        event['invalid_json'] = True
                    save(event)
                if observer in ready:
                    chunk = observer.recv(65536)
                    if not chunk:
                        raise ConnectionError('BLE provider disconnected; recording is incomplete')
                    buffer += chunk
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        event = json.loads(line)
                        if event.get('provider') == 'observing':
                            continue
                        save(event)
                        if event.get('type') == 'recorder_overflow':
                            raise RuntimeError('BLE event queue overflow; recording is incomplete')
            complete = True
        except KeyboardInterrupt:
            complete = True
        finally:
            if subscription:
                try: udp.sendto(b'POSE_UNSUBSCRIBE', subscription)
                except OSError: pass
            observer.close()
            save(dict(type='session_end', complete=complete,
                      counts=dict(counts),
                      monotonic_ns=time.monotonic_ns(), wall_time_ns=time.time_ns()))
    print('Recorded', count, 'events.')
    if not counts.get('udp'):
        print('Warning: no UDP telemetry was received.', file=sys.stderr)


if __name__ == '__main__':
    main()
