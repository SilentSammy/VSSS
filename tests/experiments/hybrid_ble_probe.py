"""Verify that the COM18 hybrid firmware serves BLE while receiving UDP poses."""
import argparse
import asyncio
import statistics
import struct
import time

from bleak import BleakClient, BleakScanner

COMMAND_UUID = "12345678-1234-5678-1234-56789abcdef1"
POSE_UUID = "12345678-1234-5678-1234-56789abcdef4"
STATS_UUID = "12345678-1234-5678-1234-56789abcdef5"


async def probe(name, duration, address_type=None):
    latencies = []
    samples = []
    options = {} if address_type is None else {"winrt": {"address_type": address_type}}
    # Keep WinRT discovery alive while creating the GATT object. Some Windows
    # adapters drop a never-paired device as soon as their watcher stops.
    async with BleakScanner() as scanner:
        deadline = time.monotonic() + 15
        device = None
        while device is None and time.monotonic() < deadline:
            device = next((item for item in scanner.discovered_devices
                           if item.name == name), None)
            if device is None:
                await asyncio.sleep(.1)
        if device is None:
            raise RuntimeError("BLE device %r was not found" % name)
        await asyncio.sleep(2)  # Allow WinRT's unpaired-device cache to settle.
        client = BleakClient(device, **options)
        await client.connect()
    try:
        started = time.monotonic()
        while time.monotonic() - started < duration:
            before = time.perf_counter()
            await client.write_gatt_char(COMMAND_UUID, bytes(3), response=True)
            latencies.append((time.perf_counter() - before) * 1000)
            pose = struct.unpack("<fff", await client.read_gatt_char(POSE_UUID))
            packets, errors = struct.unpack("<II", await client.read_gatt_char(STATS_UUID))
            samples.append((time.monotonic() - started, pose, packets, errors))
            await asyncio.sleep(.1)
    finally:
        await client.disconnect()
    gaps = [b[3] - a[3] for a, b in zip(samples, samples[1:])]
    result = {
        "samples": len(samples),
        "ble_write_ms_median": statistics.median(latencies),
        "ble_write_ms_p95": sorted(latencies)[int(.95 * (len(latencies) - 1))],
        "udp_packets_during_probe": samples[-1][2] - samples[0][2],
        "udp_parse_errors": samples[-1][3],
        "largest_packet_step": max(gaps, default=0),
        "last_pose": samples[-1][1],
    }
    for key, value in result.items():
        print("%s: %s" % (key, value))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="car18-test")
    parser.add_argument("--duration", type=float, default=15)
    parser.add_argument("--address-type", choices=("public", "random"))
    args = parser.parse_args()
    asyncio.run(probe(args.name, args.duration, args.address_type))


if __name__ == "__main__":
    main()
