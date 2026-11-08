"""Compare fresh-process BLE discovery, connection, and first-command time."""
import argparse
import asyncio
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import os
import subprocess
import sys
import time
from bleak import BleakClient, BleakScanner

SERVICE = '12345678-1234-5678-1234-56789abcdef0'
CONTROL = '12345678-1234-5678-1234-56789abcdef1'
NAME = 'VSSS_LED_TEST'


async def worker(mode, address, device_override=None, abrupt=False):
    result = {'mode': mode, 'success': False}
    client = None
    started = time.perf_counter()
    first_seen = None

    def detected(device, advertisement):
        nonlocal first_seen
        if advertisement.local_name == NAME and first_seen is None:
            first_seen = time.perf_counter() - started

    try:
        if device_override is not None:
            device = device_override
        elif mode == 'baseline':
            devices = await BleakScanner.discover(timeout=10, detection_callback=detected)
            device = next((d for d in devices if d.name == NAME), None)
        elif mode == 'known_address':
            device = None
        else:
            device = await BleakScanner.find_device_by_name(NAME, timeout=10, detection_callback=detected)
        found = time.perf_counter()
        if device is None and mode != 'known_address':
            raise RuntimeError('Device not advertising or not found')
        options = {}
        if mode in ('service_filter', 'cached_services'):
            options['services'] = [SERVICE]
        if mode in ('cached_services', 'cached_all'):
            options['winrt'] = {'use_cached_services': True}
        if mode == 'uncached_services':
            options['winrt'] = {'use_cached_services': False}
        target = address if mode == 'known_address' else (
            device.address if mode in ('baseline', 'name_address') else device)
        client = BleakClient(target, timeout=10, pair=False, **options)
        await client.connect()
        connected = time.perf_counter()
        await client.write_gatt_char(CONTROL, b'\x01', response=True)
        written = time.perf_counter()
        state = await client.read_gatt_char(CONTROL)
        if state != b'\x01':
            raise RuntimeError('LED readback mismatch')
        verified = time.perf_counter()
        await client.write_gatt_char(CONTROL, b'\x00', response=True)
        before_disconnect = time.perf_counter()
        if abrupt:
            result.update(success=True, ready_s=written - started,
                          verified_s=verified - started, abrupt_exit=True)
            print(json.dumps(result), flush=True)
            os._exit(0)
        # Informational Windows status; no pairing or unpairing is performed.
        requester = getattr(client._backend, '_requester', None)
        result['is_paired'] = requester.device_information.pairing.is_paired if requester else None
        await client.disconnect()
        disconnected = time.perf_counter()
        result.update(success=True, first_seen_s=first_seen,
                      discovery_s=found - started, connect_s=connected - found,
                      first_write_s=written - connected,
                      ready_s=written - started, verified_s=verified - started,
                      disconnect_s=disconnected - before_disconnect)
    except Exception as error:
        result.update(error=repr(error), elapsed_s=time.perf_counter() - started)
    finally:
        if client and client.is_connected:
            await client.disconnect()
    return result


def read_until(serial, prefix, timeout=15):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        line = serial.readline().decode(errors='replace').strip()
        if line.startswith(prefix):
            return line
    raise TimeoutError('No device response: ' + prefix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['baseline', 'name_address', 'name_device', 'known_address',
                                           'service_filter', 'cached_services', 'cached_all', 'uncached_services'])
    parser.add_argument('--address', default='88:56:A6:62:7E:A2')
    parser.add_argument('--abrupt', action='store_true')
    parser.add_argument('--serial', default='COM17')
    parser.add_argument('--output', default=str(Path(__file__).with_name('results.json')))
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(asyncio.run(worker(args.worker, args.address, abrupt=args.abrupt))))
        return
    from mpremote.transport_serial import SerialTransport
    transport = SerialTransport(args.serial)
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'platform': platform.platform(), 'bleak_version': importlib.metadata.version('bleak'),
              'pair_requested': False, 'samples': []}
    try:
        transport.enter_raw_repl()
        transport.exec_raw_no_follow(Path(__file__).with_name('esp32_probe.py').read_text())
        transport.serial.timeout = .1
        read_until(transport.serial, 'READY')
        for interval, modes, repetitions in [
            (100000, ['baseline'], 3),
            (100000, ['name_address', 'name_device', 'known_address', 'service_filter', 'cached_services'], 5),
            (20000, ['name_device', 'service_filter'], 8),
            (100000, ['name_device'], 5),
        ]:
            transport.serial.write(('A %d\n' % interval).encode())
            read_until(transport.serial, 'ADVERTISING')
            for trial in range(repetitions):
                for mode in modes:
                    started = time.perf_counter()
                    process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', mode,
                                              '--address', args.address], capture_output=True, text=True, timeout=40)
                    elapsed = time.perf_counter() - started
                    if process.returncode:
                        result = {'mode': mode, 'success': False, 'error': process.stderr}
                    else:
                        result = json.loads(process.stdout.strip().splitlines()[-1])
                    result.update(advertising_ms=interval / 1000, trial=trial + 1,
                                  process_wall_s=elapsed)
                    report['samples'].append(result)
                    Path(args.output).write_text(json.dumps(report, indent=2))
                    print(json.dumps(result), flush=True)
    finally:
        transport.serial.write(b'\x03')
        time.sleep(.2)
        transport.close()


if __name__ == '__main__':
    main()
