"""Measure acknowledged BLE writes and estimated one-way GPIO8 latency over USB.

Run: python ble_latency_test.py --serial COM17
The device program executes in RAM; no boot files are replaced.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import struct
import time
from bleak import BleakClient, BleakScanner
from mpremote.transport_serial import SerialTransport

UUID = '12345678-1234-5678-1234-56789abcdef1'


def read_line(serial, prefix, timeout=10):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        line = serial.readline().decode(errors='replace').strip()
        ended = time.perf_counter()
        if line.startswith(prefix):
            return line.split(), ended
        if 'Traceback' in line:
            raise RuntimeError('Device exception: ' + line)
    raise TimeoutError('No serial response: ' + prefix)


def synchronize(serial):
    samples = []
    for _ in range(30):
        started = time.perf_counter()
        serial.write(b'S\n')
        fields, ended = read_line(serial, 'SYNC', 2)
        device = int(fields[1]) / 1e6
        low, high = device - ended, device - started
        samples.append((high - low, (low + high) / 2, ended))
    width, offset, epoch = min(samples)
    return {'offset': offset, 'uncertainty_ms': width * 500, 'epoch': epoch}


def summary(values):
    if not values:
        return None
    ordered = sorted(values)
    return {'min': min(values), 'median': statistics.median(values),
            'mean': statistics.mean(values), 'p95': ordered[math.ceil(len(values) * .95) - 1],
            'max': max(values)}


async def measure(args, serial):
    print('Scanning for VSSS_LED_TEST...', flush=True)
    device = await BleakScanner.find_device_by_name('VSSS_LED_TEST', timeout=15)
    if device is None:
        raise RuntimeError('BLE test device not found')
    results = []
    async with BleakClient(device) as client:
        print('Connected:', device.address, flush=True)
        for trial in range(args.trials):
            before = await asyncio.to_thread(synchronize, serial)
            sent, writes = {}, []
            for seq in range(args.count):
                state = (seq + 1) % 2
                packet = bytes([state]) + struct.pack('<I', seq)
                started = time.perf_counter()
                sent[seq] = started
                await client.write_gatt_char(UUID, packet, response=True)
                ended = time.perf_counter()
                writes.append((ended - started) * 1000)
                await asyncio.sleep(max(0, args.interval - (ended - started)))
            await asyncio.sleep(.2)
            after = await asyncio.to_thread(synchronize, serial)
            serial.write(b'D\n')
            records = {}
            deadline = time.perf_counter() + 20
            while time.perf_counter() < deadline:
                fields = serial.readline().decode(errors='replace').strip().split()
                if fields == ['DONE']:
                    break
                if len(fields) == 4 and fields[0] == 'ACK':
                    records[int(fields[1])] = (int(fields[2]) / 1e6, int(fields[3]) / 1e6)
            else:
                raise TimeoutError('Incomplete USB timestamp download')
            samples = []
            for seq, (rx, applied) in records.items():
                if seq not in sent:
                    raise RuntimeError('Unexpected command sequence')
                fraction = min(1, max(0, (sent[seq] - before['epoch']) /
                                     (after['epoch'] - before['epoch'])))
                offset = before['offset'] + fraction * (after['offset'] - before['offset'])
                samples.append({'seq': seq, 'receive_ms': (rx - sent[seq] - offset) * 1000,
                                'led_ms': (applied - sent[seq] - offset) * 1000})
            times = list(sent.values())
            result = {'trial': trial + 1, 'sent': args.count, 'received': len(records),
                      'actual_send_interval_ms': summary([(b - a) * 1000 for a, b in zip(times, times[1:])]),
                      'write_completion_ms': summary(writes),
                      'one_way_receive_ms': summary([s['receive_ms'] for s in samples]),
                      'one_way_led_ms': summary([s['led_ms'] for s in samples]),
                      'clock_uncertainty_ms': [before['uncertainty_ms'], after['uncertainty_ms']],
                      'clock_offset_change_ms': (after['offset'] - before['offset']) * 1000,
                      'samples': samples}
            results.append(result)
            save_report(args.output, results)
            print(json.dumps({k: v for k, v in result.items() if k != 'samples'}, indent=2), flush=True)
    return results


def save_report(path, results):
    Path(path).write_text(json.dumps({
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'method': 'USB midpoint clock synchronization before/after each trial; linear drift interpolation; '
                  'device buffers BLE receive and GPIO update timestamps; acknowledged GATT writes; '
                  'one-way times exclude USB reporting but include host/BLE stack scheduling.',
        'trials': results}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial', default='COM17')
    parser.add_argument('--count', type=int, default=500)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--interval', type=float, default=.02)
    parser.add_argument('--output', default='ble_latency_results.json')
    args = parser.parse_args()
    if not 1 <= args.count <= 1000 or args.trials < 1 or args.interval <= 0:
        parser.error('count must be 1..1000, trials and interval must be positive')
    if args.trials * args.count * max(args.interval, .1) > 400:
        parser.error('Choose a shorter test to avoid the device timestamp wrap limit')
    transport = SerialTransport(args.serial)
    try:
        transport.enter_raw_repl()
        transport.exec_raw_no_follow((Path(__file__).parent / 'mp/ble_latency_test.py').read_text())
        transport.serial.timeout = .1
        read_line(transport.serial, 'READY')
        results = asyncio.run(measure(args, transport.serial))
        save_report(args.output, results)
        print('Saved', args.output, flush=True)
    except Exception:
        if transport.serial.in_waiting:
            print('Device output:', transport.serial.read(transport.serial.in_waiting).decode(errors='replace'), flush=True)
        raise
    finally:
        transport.serial.write(b'\x03')
        time.sleep(.2)
        transport.close()


if __name__ == '__main__':
    main()
