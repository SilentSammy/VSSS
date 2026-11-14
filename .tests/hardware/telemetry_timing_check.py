"""Run via mpremote; stationary Wi-Fi/BLE timing A/B, never creates motors.

Connect a Car client during the BLE wait. Velocity/waypoint writes are ignored.
Results are printed and saved as telemetry_timing_check.json on the board.
The normal main firmware is restored by a reset after this diagnostic ends.
"""
import json
import struct
import time

import main as firmware
from robot_config import RobotConfig
from pose_receiver import PoseReceiver


def statistics(values):
    if not values:
        return None
    ordered = sorted(values)
    return dict(count=len(values), min=min(values), max=max(values),
                mean=sum(values) / len(values), median=ordered[len(ordered) // 2],
                p95=ordered[min(len(ordered) - 1, int(len(ordered) * .95))])


settings = RobotConfig().data
wifi = firmware.connect_wifi(settings['wifi_ssid'], settings['wifi_password'])
if not wifi.isconnected():
    raise RuntimeError('Wi-Fi unavailable for stationary timing test')

import ble_server
ble_server.DEVICE_NAME = 'car%d' % settings['car_id']
ble_server.disconnect_callback = lambda: None
ble_server.control_callbacks = {
    firmware.VELOCITY_UUID: lambda data: bytes(3),
    firmware.LEASE_UUID: lambda data: bytes(1),
    firmware.WAYPOINT_UUID: lambda data: bytes(12),
    firmware.TRIM_MATRIX_UUID: lambda data: None,
    firmware.CONTROLLER_UUID: lambda data: None,
    firmware.STATUS_UUID: lambda data: None,
    firmware.CONTROL_UUID: lambda data: bytes(2),
}
ble_server.initial_values = {
    firmware.VELOCITY_UUID: bytes(3), firmware.LEASE_UUID: bytes(1),
    firmware.WAYPOINT_UUID: bytes(12), firmware.CONTROL_UUID: bytes(2),
    firmware.TRIM_MATRIX_UUID: struct.pack('<9f', *settings['trim_matrix']),
    firmware.CONTROLLER_UUID: firmware.controller_payload(settings['controller']),
    firmware.STATUS_UUID: bytes(20),
}
receiver = None
results = []
try:
    ble_server.start()
    print('STATIONARY TEST: waiting for BLE client; all motion writes ignored')
    deadline = time.ticks_add(time.ticks_ms(), 25000)
    while not ble_server._connections and time.ticks_diff(deadline, time.ticks_ms()) > 0:
        ble_server.poll()
        time.sleep_ms(10)
    if not ble_server._connections:
        raise RuntimeError('No BLE client; cannot measure coexistence timing')
    receiver = PoseReceiver(settings['car_id'], server=settings['telemetry_host'],
                            port=settings['telemetry_port'])
    for mode in (wifi.PM_PERFORMANCE, wifi.PM_NONE):
        wifi.config(pm=mode)
        start = time.ticks_ms()
        last_packet = last_frame = last_sequence = None
        packet_gaps, frame_gaps, frame_ages, batches = [], [], [], []
        packet_total = 0
        status_at = start
        while time.ticks_diff(time.ticks_ms(), start) < 12000:
            ble_server.poll()
            before = receiver.packets
            receiver.poll()
            now = time.ticks_ms()
            if time.ticks_diff(now, start) >= 2000:
                delta = receiver.packets - before
                if delta:
                    packet_total += delta
                    batches.append(delta)
                    if last_packet is not None:
                        packet_gaps.append(time.ticks_diff(now, last_packet))
                    last_packet = now
                identity = receiver.stream_id, receiver.frame_sequence
                if receiver.received_ms is not None and identity != last_sequence:
                    if last_frame is not None:
                        frame_gaps.append(time.ticks_diff(now, last_frame))
                    last_frame, last_sequence = now, identity
                    if receiver.frame_age_ms is not None:
                        frame_ages.append(receiver.frame_age_ms)
            if time.ticks_diff(now, status_at) >= 100:
                status_at = now
                pose = receiver.pose or (0., 0., 0.)
                ble_server.set_value(firmware.STATUS_UUID, struct.pack('<BBBBfffI',
                    0, 1 | (2 if receiver.fresh else 0), 0, 0, *pose, receiver.packets))
            time.sleep_ms(10)
        result = dict(pm=mode, ble_connected=bool(ble_server._connections),
                      packets=packet_total, receive_gap_ms=statistics(packet_gaps),
                      frame_gap_ms=statistics(frame_gaps), frame_age_ms=statistics(frame_ages),
                      maximum_packet_batch=max(batches) if batches else 0,
                      packet_gaps_ms=packet_gaps)
        results.append(result)
        print(json.dumps(result))
finally:
    wifi.config(pm=wifi.PM_NONE)
    if receiver is not None:
        receiver.close()
    ble_server.stop()
    with open('telemetry_timing_check.json', 'w') as output:
        json.dump(results, output)
