"""Fresh-process integration checks of the production MecanumBLEClient."""
import argparse
import asyncio
import json
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def worker():
    sys.path.insert(0, str(ROOT))
    from mecanum_client import MecanumBLEClient
    client = MecanumBLEClient(device_name='VSSS_LED_TEST')
    started = time.perf_counter()
    try:
        client.connect()
        while not client.is_connected and time.perf_counter() - started < 15:
            time.sleep(.005)
        if not client.is_connected:
            raise TimeoutError('Production client did not connect')
        connected = time.perf_counter()
        async def verify():
            await client.client.write_gatt_char(client.uuid_velocity, b'\x01', response=True)
            if await client.client.read_gatt_char(client.uuid_velocity) != b'\x01':
                raise RuntimeError('LED readback mismatch')
            await client.client.write_gatt_char(client.uuid_velocity, b'\x00', response=True)
        asyncio.run_coroutine_threadsafe(verify(), client.loop).result(timeout=10)
        verified = time.perf_counter()
        print(json.dumps({'success': True, 'connected_s': connected - started,
                          'verified_on_off_s': verified - started,
                          'bleak_version': importlib.metadata.version('bleak')}), flush=True)
    finally:
        client.disconnect()
        client.loop.call_soon_threadsafe(client.loop.stop)
        client.thread.join(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--output', default=str(Path(__file__).with_name('client_results.json')))
    args = parser.parse_args()
    if args.worker:
        sys.stdout.reconfigure(encoding='utf-8')
        worker()
        return
    from mpremote.transport_serial import SerialTransport
    transport = SerialTransport('COM17')
    results = []
    try:
        transport.enter_raw_repl()
        transport.exec_raw_no_follow((ROOT / 'mp/ble_led_test.py').read_text())
        transport.serial.timeout = .1
        deadline = time.perf_counter() + 10
        while time.perf_counter() < deadline:
            if b'GPIO8 BLE LED ready' in transport.serial.readline():
                break
        else:
            raise RuntimeError('Device test did not start')
        for trial in range(5):
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'],
                                     capture_output=True, text=True, encoding='utf-8', timeout=30)
            records = [line for line in process.stdout.splitlines() if line.startswith('{')]
            result = json.loads(records[-1]) if process.returncode == 0 and records else {
                'success': False, 'output': process.stdout, 'error': process.stderr}
            result['trial'] = trial + 1
            results.append(result)
            Path(args.output).write_text(json.dumps(results, indent=2))
            print(json.dumps(result), flush=True)
    finally:
        transport.serial.write(b'\x03')
        time.sleep(.2)
        transport.close()


if __name__ == '__main__':
    main()
