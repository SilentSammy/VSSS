"""Bench checks for an unmounted robot board. Sends nonzero velocities."""
import asyncio
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bleak import BleakClient, BleakScanner
from mecanum_client import MecanumBLEClient

NAME = 'Eq4'
UUID = '12345678-1234-5678-1234-56789abcdef1'


async def discover():
    started = time.monotonic()
    device = await BleakScanner.find_device_by_name(NAME, timeout=6)
    if device is None:
        raise RuntimeError('Robot did not resume advertising')
    return time.monotonic() - started


async def firmware_timeout():
    device = await BleakScanner.find_device_by_name(NAME, timeout=6)
    if device is None:
        raise RuntimeError('Robot not found')
    disconnected = asyncio.Event()
    client = BleakClient(device, disconnected_callback=lambda _: disconnected.set())
    try:
        await client.connect()
        await client.write_gatt_char(UUID, bytes([16, 0, 0]), response=True)
        started = time.monotonic()
        # This client deliberately sends no heartbeats.
        await asyncio.wait_for(disconnected.wait(), timeout=4)
        return time.monotonic() - started
    finally:
        await client.disconnect()


def main():
    results = {}
    destination = Path(__file__).with_name('hardware_results.json')

    def save(name, value):
        results[name] = value
        destination.write_text(json.dumps(results, indent=2))
        print(name, json.dumps(value), flush=True)

    for trial in range(3):
        robot = MecanumBLEClient(NAME)
        started = time.monotonic()
        try:
            robot.connect()
            robot.wait_until_connected(15)
            ready = time.monotonic() - started
            robot.set_velocity({'x': .2, 'y': -.1, 'w': .1})
            time.sleep(2.5)  # Longer than the firmware lease: heartbeat must work.
            if not robot.is_connected:
                raise RuntimeError('Connection lost despite worker heartbeat')
            robot.stop()
            closed = time.monotonic()
            robot.disconnect()
            save('normal_%d' % (trial + 1), dict(ready_s=ready,
                 disconnect_s=time.monotonic() - closed, heartbeat_held_connection=True))
        finally:
            robot.disconnect()

    save('no_heartbeat_disconnect_s', asyncio.run(firmware_timeout()))
    save('discovery_after_timeout_s', asyncio.run(discover()))

    code = """
import os, time
from mecanum_client import MecanumBLEClient
car = MecanumBLEClient('Eq4')
car.connect()
car.wait_until_connected(15)
car.set_velocity({'x': .2})
time.sleep(.2)
os._exit(0)
"""
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True, timeout=25)
    save('discovery_after_script_crash_s', asyncio.run(discover()))

    robot = MecanumBLEClient(NAME)
    try:
        robot.connect()
        robot.wait_until_connected(15)
        robot.set_velocity({'x': .2})
        time.sleep(.2)
        robot._process.kill()  # Deliberately bypass worker cleanup.
        robot._process.wait(timeout=3)
        save('discovery_after_worker_killed_s', asyncio.run(discover()))
    finally:
        robot.disconnect()


if __name__ == '__main__':
    main()
