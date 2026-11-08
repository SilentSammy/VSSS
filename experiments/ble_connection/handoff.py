"""Test same-process device reuse and immediate reconnect after abrupt exits."""
import asyncio
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from bleak import BleakScanner
from mpremote.transport_serial import SerialTransport
from benchmark import NAME, worker, read_until


async def reused_device():
    device = await BleakScanner.find_device_by_name(NAME, timeout=10)
    if device is None:
        raise RuntimeError('Test device not found')
    results = []
    for trial in range(5):
        result = await worker('name_device', device.address, device_override=device)
        result.update(scenario='reuse_device_same_process', trial=trial + 1)
        results.append(result)
        print(json.dumps(result), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watchdog-ms', type=int, default=0)
    args = parser.parse_args()
    transport = SerialTransport('COM17')
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'advertising_ms': 20,
              'watchdog_ms': args.watchdog_ms, 'samples': []}
    destination = Path(__file__).with_name('handoff_watchdog_results.json' if args.watchdog_ms else 'handoff_results.json')
    try:
        transport.enter_raw_repl()
        transport.exec_raw_no_follow(Path(__file__).with_name('esp32_probe.py').read_text())
        transport.serial.timeout = .1
        read_until(transport.serial, 'READY')
        transport.serial.write(b'A 20000\n')
        read_until(transport.serial, 'ADVERTISING')
        if args.watchdog_ms:
            transport.serial.write(('W %d\n' % args.watchdog_ms).encode())
            read_until(transport.serial, 'WATCHDOG')
        # Keep every owner in a fresh process, matching student script restarts.
        for trial in range(5):
            for abrupt in (True, False):
                command = [sys.executable, str(Path(__file__).with_name('benchmark.py')), '--worker', 'name_device']
                if abrupt:
                    command.append('--abrupt')
                started = time.perf_counter()
                process = subprocess.run(command, capture_output=True, text=True, timeout=30)
                result = json.loads(process.stdout.strip().splitlines()[-1]) if process.returncode == 0 else {
                    'success': False, 'error': process.stderr}
                result.update(scenario='abrupt_exit' if abrupt else 'restart_after_abrupt_exit',
                              trial=trial + 1, process_wall_s=time.perf_counter() - started)
                report['samples'].append(result)
                destination.write_text(json.dumps(report, indent=2))
                print(json.dumps(result), flush=True)
    finally:
        transport.serial.write(b'\x03')
        time.sleep(.2)
        transport.close()


if __name__ == '__main__':
    main()
