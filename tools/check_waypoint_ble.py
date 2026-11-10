"""Safely exercise waypoint mode by targeting the robot's current pose."""

import asyncio
import struct

from bleak import BleakClient, BleakScanner


WAYPOINT_UUID = "12345678-1234-5678-1234-56789abcdef6"
LEASE_UUID = "12345678-1234-5678-1234-56789abcdef2"
CONTROL_UUID = "12345678-1234-5678-1234-56789abcdef7"
STATUS_UUID = "12345678-1234-5678-1234-56789abcdef9"
STATUS = struct.Struct("<BBBBfffI")


async def main():
    device = await BleakScanner.find_device_by_name("car1", timeout=10)
    if device is None:
        raise RuntimeError("car1 not found")
    async with BleakClient(device, timeout=10) as client:
        initial = STATUS.unpack(await client.read_gatt_char(STATUS_UUID))
        print("initial", initial)
        mode, flags, fault, _, x, y, angle, _ = initial
        if flags & 2 == 0:
            raise RuntimeError("car1 has no fresh onboard pose; waypoint not sent")
        await client.write_gatt_char(
            WAYPOINT_UUID, struct.pack("<fff", x, y, angle), response=True)
        for _ in range(8):
            await asyncio.sleep(.2)
            await client.write_gatt_char(LEASE_UUID, bytes(1), response=True)
            print("status", STATUS.unpack(await client.read_gatt_char(STATUS_UUID)))
        await client.write_gatt_char(CONTROL_UUID, bytes(1), response=True)


asyncio.run(main())
