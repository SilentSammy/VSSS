"""Compare observed ball control and recorded BLE commands in lab bags."""
import argparse
from bisect import bisect_right
from collections import Counter
import json
import math
import struct
from pathlib import Path
from statistics import median

from .ball_dynamics import load_samples


def analyze(path, car_id=1):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    samples = load_samples(Path(path), car_id)
    writes = sorted((row for row in rows if row['type'] == 'ble' and row.get('car') == f'car{car_id}'
                     and row.get('phase') == 'start' and row.get('operation') == 'write'
                     and row.get('characteristic') in ('velocity', 'waypoint')),
                    key=lambda row: row['wall_time_ns'])
    times = [row['wall_time_ns'] / 1e9 for row in writes]
    near = []
    paused = forward = 0
    for point in samples:
        index = bisect_right(times, point.time) - 1
        if index < 0 or writes[index]['characteristic'] != 'velocity':
            continue
        command = writes[index].get('value')
        if not isinstance(command, dict):
            try:
                command = dict(zip(('x', 'y', 'w'), (v / 128 for v in struct.unpack(
                    '<bbb', bytes.fromhex(writes[index].get('payload_hex', ''))))))
            except (ValueError, struct.error):
                command = {}
        x, y, w = (command.get(axis, 0.) for axis in ('x', 'y', 'w'))
        ball_speed = math.hypot(point.ball_vx, point.ball_vy)
        paused += x == 0 and y == 0 and w != 0 and ball_speed > .03
        if x > .05:
            forward += 1
            if math.hypot(point.car_x - point.ball_x, point.car_y - point.ball_y) < .18 and ball_speed > .03:
                near.append((math.hypot(point.car_vx, point.car_vy), ball_speed, x))
    packets = [row['packet'] for row in rows if row['type'] == 'udp' and row.get('valid')]
    starts, durations = {}, []
    for row in rows:
        if row['type'] != 'ble':
            continue
        key = row.get('connection_id'), row.get('request')
        if row.get('phase') == 'start':
            starts[key] = row
        elif row.get('phase') == 'complete' and key in starts:
            start = starts.pop(key)
            if start.get('operation') == 'write' and start.get('characteristic') in ('velocity', 'waypoint'):
                durations.append((row['monotonic_ns'] - start['monotonic_ns']) / 1e6)
    ages = [packet['frame_age_ms'] for packet in packets if packet.get('frame_age_ms') is not None]
    snapshot = next((row.get('provider', row) for row in rows
                     if row['type'] in ('bag_header', 'provider_snapshot')), {})
    setup = (((snapshot.get('vision') or {}).get('metadata') or {}).get('status') or {}).get('setup') or {}
    field = (setup.get('board') or {}).get('dimensions_m')
    outside = [point for packet in packets for point in packet.get('balls', [])
               if field and any(abs(point[axis]) > size / 2 for axis, size in zip(('x', 'y'), field))]
    return dict(source=str(path), event_counts=dict(Counter(row['type'] for row in rows)),
                correlated_samples=len(samples), duration_s=samples[-1].time - samples[0].time,
                recorded_waypoints=sum(row['characteristic'] == 'waypoint' for row in writes),
                forward_samples=forward, rotation_only_while_ball_moves_samples=paused,
                near_forward_samples=len(near),
                near_car_speed_median_mps=median(row[0] for row in near) if near else None,
                near_ball_speed_median_mps=median(row[1] for row in near) if near else None,
                near_forward_command_median=median(row[2] for row in near) if near else None,
                no_ball_frames=sum(not packet.get('balls') for packet in packets),
                multiple_ball_frames=sum(len(packet.get('balls', [])) > 1 for packet in packets),
                camera_age_median_ms=median(ages) if ages else None,
                ble_write_duration_median_ms=median(durations) if durations else None,
                first_ball=[samples[0].ball_x, samples[0].ball_y],
                last_ball=[samples[-1].ball_x, samples[-1].ball_y],
                field_dimensions_m=field,
                raw_outside_ball_detections=len(outside) if field else None,
                definitions='Near-forward: positive body x, car/ball gap < 0.18 m, ball speed > 0.03 m/s. '
                            'Speeds use five-frame local slopes. Sample counts are not matched trial durations; '
                            'nearby motion does not prove contact. Raw outside detections may be outliers, not physical exits. '
                            'GATT completion time is not one-way actuation latency.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bags', nargs='+', type=Path)
    parser.add_argument('--car', type=int, default=1)
    parser.add_argument('--output', type=Path, help='write comparison to a new JSON file')
    args = parser.parse_args(argv)
    reports = [analyze(path, args.car) for path in args.bags]
    text = json.dumps(reports, indent=2)
    print(text)
    if args.output:
        with args.output.open('x', encoding='utf-8') as destination:
            destination.write(text + '\n')


if __name__ == '__main__':
    main()
