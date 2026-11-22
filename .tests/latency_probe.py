"""Measure IPC separately from BLE. Hardware measurements send STOP only."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vsss import CarCmd, EventHub, GPSState, VSSSProvider, Velocity, rpc_method


class NoHardware(CarCmd):
    @rpc_method
    def available(self, refresh=False): return [1]
    def _apply(self, car_id, velocity): pass


def measure(function, count):
    values = []
    for _ in range(count):
        start = time.perf_counter_ns()
        function()
        values.append((time.perf_counter_ns() - start) / 1e6)
    ordered = sorted(values)
    return dict(samples=count, median_ms=round(statistics.median(values), 3),
                p95_ms=round(ordered[max(0, int(.95 * count + .999) - 1)], 3),
                min_ms=round(ordered[0], 3), max_ms=round(ordered[-1], 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=200)
    parser.add_argument('--endpoint', help='Also measure read-only calls to a running provider')
    parser.add_argument('--stop-only', action='store_true', help='Also measure acknowledged STOP on the running provider')
    parser.add_argument('--car', type=int, default=1)
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    if args.worker:
        events = EventHub()
        with VSSSProvider(GPSState(events), NoHardware(events)) as provider, \
             provider.serve(endpoint=args.worker, port=0):
            print('ready', flush=True)
            sys.stdin.readline()
        return
    report = {}
    with tempfile.TemporaryDirectory() as directory:
        endpoint = Path(directory) / 'rpc.json'
        process = subprocess.Popen([sys.executable, __file__, '--worker', str(endpoint)],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            if process.stdout.readline().strip() != 'ready':
                raise RuntimeError(process.stderr.read())
            with VSSSProvider.connect(endpoint) as client:
                report['cross_process_rpc_no_hardware'] = measure(lambda: client.cars.stop(1), args.samples)
        finally:
            process.communicate('\n', timeout=10)
    if args.endpoint or args.stop_only:
        with VSSSProvider.connect(args.endpoint) as client:
            report['running_provider_read_only_rpc'] = measure(client.describe, args.samples)
            if args.stop_only:
                if any(command.value != Velocity() for command in client.cars.observed()):
                    raise RuntimeError('A nonzero command is active; hardware benchmark refused')
                # Keep the explicitly requested hardware sample brief.
                report['acknowledged_stop_including_device'] = measure(lambda: client.cars.stop(args.car), min(args.samples, 20))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
