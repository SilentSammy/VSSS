"""Portable UDP LED RTT benchmark, suitable for the hotspot host or Windows."""
import argparse
from datetime import datetime, timezone
import json
import math
import platform
import socket
import statistics
import time
from pathlib import Path


def stats(values):
    if not values:
        return None
    ordered = sorted(values)
    return dict(min=min(values), median=statistics.median(values), mean=statistics.mean(values),
                p95=ordered[math.ceil(len(ordered) * .95) - 1],
                p99=ordered[math.ceil(len(ordered) * .99) - 1], max=max(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ip')
    parser.add_argument('--count', type=int, default=500)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--interval', type=float, default=.02)
    parser.add_argument('--timeout', type=float, default=.5)
    parser.add_argument('--output', default='udp_roundtrip_results.json')
    args = parser.parse_args()
    if args.count < 1 or args.trials < 1 or args.interval < 0 or args.timeout <= 0:
        parser.error('Invalid count, trials, interval, or timeout')
    report = {'host': platform.node(), 'platform': platform.platform(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'target': args.ip, 'target_interval_ms': args.interval * 1000,
              'timeout_ms': args.timeout * 1000, 'trials': []}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
        udp.connect((args.ip, 5002))
        report['source_ip'] = udp.getsockname()[0]
        try:
            for trial in range(args.trials):
                samples, processing, sends = [], [], []
                for index in range(args.count):
                    seq = trial * args.count + index
                    state = (index + 1) % 2
                    payload = json.dumps({'seq': seq, 'led': state}).encode()
                    started = time.perf_counter()
                    sends.append(started)
                    udp.send(payload)
                    deadline = started + args.timeout
                    while time.perf_counter() < deadline:
                        udp.settimeout(max(.0001, deadline - time.perf_counter()))
                        try:
                            data = udp.recv(256)
                        except socket.timeout:
                            break
                        ended = time.perf_counter()
                        try:
                            ack = json.loads(data)
                            if ack.get('seq') != seq or ack.get('led') != state:
                                continue
                            samples.append((ended - started) * 1000)
                            processing.append(ack['processing_us'])
                            break
                        except (ValueError, KeyError, AttributeError):
                            continue
                    time.sleep(max(0, args.interval - (time.perf_counter() - started)))
                result = {'trial': trial + 1, 'sent': args.count, 'acknowledged': len(samples),
                          'missing_replies': args.count - len(samples), 'rtt_ms': stats(samples),
                          'device_processing_us': stats(processing),
                          'actual_send_interval_ms': stats([(b - a) * 1000 for a, b in zip(sends, sends[1:])]),
                          'samples_ms': samples}
                report['trials'].append(result)
                Path(args.output).write_text(json.dumps(report, indent=2))
                print(json.dumps({k: v for k, v in result.items() if k != 'samples_ms'}), flush=True)
        finally:
            for _ in range(3):
                udp.send(json.dumps({'seq': -1, 'led': 0}).encode())
                time.sleep(.02)
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    main()
