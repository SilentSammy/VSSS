"""RAM-only BLE GPIO8 timing test, with buffered timestamps and USB clock sync."""
from machine import Pin
import network
import time
import select
import sys
import struct
import gc
import ble_server


def main():
    network.WLAN(network.STA_IF).active(False)
    network.WLAN(network.AP_IF).active(False)
    led = Pin(8, Pin.OUT, value=1)
    records = bytearray(1000 * 12)
    record_count = 0
    origin = time.ticks_us()
    uuid = '12345678-1234-5678-1234-56789abcdef1'

    def receive(data):
        nonlocal record_count
        received = time.ticks_diff(time.ticks_us(), origin)
        if len(data) == 5 and data[0] in (0, 1):
            led.value(1 - data[0])
            applied = time.ticks_diff(time.ticks_us(), origin)
            seq = struct.unpack('<I', data[1:])[0]
            if record_count < 1000:
                struct.pack_into('<III', records, record_count * 12, seq, received, applied)
                record_count += 1

    ble_server.DEVICE_NAME = 'VSSS_LED_TEST'
    ble_server.control_callbacks = {uuid: receive}
    poll = select.poll()
    poll.register(sys.stdin, select.POLLIN)
    try:
        ble_server.start()
        gc.collect()
        print('READY')
        while True:
            if poll.poll(1):
                command = sys.stdin.readline().strip()
                if command == 'S':
                    print('SYNC', time.ticks_diff(time.ticks_us(), origin))
                elif command == 'D':
                    for index in range(record_count):
                        seq, received, applied = struct.unpack_from('<III', records, index * 12)
                        print('ACK', seq, received, applied)
                    record_count = 0
                    gc.collect()
                    print('DONE')
                elif command == 'Q':
                    return
    finally:
        led.value(1)
        ble_server.stop()


if __name__ == '__main__':
    main()
