"""Windows terminal BLE LED control: 1 = on, 0 = off, Q/Esc = quit."""
import argparse
import asyncio
import msvcrt
from bleak import BleakClient, BleakScanner

LED_UUID = '12345678-1234-5678-1234-56789abcdef1'


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='VSSS_LED_TEST')
    parser.add_argument('--check', action='store_true', help='verify on/off writes and readback, then exit')
    args = parser.parse_args()
    print('Scanning for %s...' % args.name, flush=True)
    device = await BleakScanner.find_device_by_name(args.name, timeout=15)
    if device is None:
        raise SystemExit('BLE LED device not found. Start mp/ble_led_test.py on the ESP32.')
    async with BleakClient(device) as client:
        print('Connected to %s (%s)' % (args.name, device.address), flush=True)
        try:
            if args.check:
                for state in (1, 0):
                    await client.write_gatt_char(LED_UUID, bytes([state]), response=True)
                    await asyncio.sleep(.1)
                    actual = await client.read_gatt_char(LED_UUID)
                    if actual != bytes([state]):
                        raise RuntimeError('LED state readback mismatch: %r' % actual)
                    print('Verified LED ' + ('ON' if state else 'OFF'), flush=True)
                return
            print('1 = ON | 0 = OFF | Q or Esc = quit (no Enter needed)', flush=True)
            while client.is_connected:
                if msvcrt.kbhit():
                    key = msvcrt.getwch().lower()
                    if key in ('q', '\x1b'):
                        break
                    if key in ('1', '0'):
                        # Match the existing car characteristic's acknowledged writes.
                        await client.write_gatt_char(LED_UUID, bytes([int(key)]), response=True)
                        print('Wrote LED ' + ('ON' if key == '1' else 'OFF'), flush=True)
                await asyncio.sleep(.001)
            if not client.is_connected:
                print('BLE connection lost. Restart the client to reconnect.', flush=True)
        finally:
            if client.is_connected:
                await client.write_gatt_char(LED_UUID, b'\x00', response=True)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
