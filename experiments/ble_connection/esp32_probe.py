"""RAM-only connection benchmark: existing BLE service, GPIO8, USB control."""
import network
import select
import sys
import time
from machine import Pin
import ble_server


def main():
    network.WLAN(network.STA_IF).active(False)
    network.WLAN(network.AP_IF).active(False)
    led = Pin(8, Pin.OUT, value=1)
    uuid = '12345678-1234-5678-1234-56789abcdef1'
    interval = 100000
    idle_disconnect_ms = 0
    last_activity = time.ticks_ms()
    received_command = False
    original_advertise = ble_server._advertise

    def advertise():
        original_advertise(interval)

    def command(data):
        nonlocal last_activity, received_command
        last_activity = time.ticks_ms()
        received_command = True
        if len(data) == 1 and data[0] in (0, 1):
            led.value(1 - data[0])
            ble_server._ble.gatts_write(ble_server._handles[uuid], data)

    ble_server._advertise = advertise
    ble_server.DEVICE_NAME = 'VSSS_LED_TEST'
    ble_server.control_callbacks = {uuid: command}
    original_irq = ble_server._ble_irq

    def irq(event, data):
        nonlocal last_activity, received_command
        if event == 1:
            last_activity = time.ticks_ms()
            received_command = False
        original_irq(event, data)

    ble_server._ble_irq = irq
    poll = select.poll()
    poll.register(sys.stdin, select.POLLIN)
    try:
        ble_server.start()
        print('READY')
        while True:
            if poll.poll(10):
                message = sys.stdin.readline().strip().split()
                if message and message[0] == 'A':
                    interval = int(message[1])
                    advertise()
                    print('ADVERTISING', interval)
                elif message == ['Q']:
                    return
                elif message and message[0] == 'W':
                    idle_disconnect_ms = int(message[1])
                    print('WATCHDOG', idle_disconnect_ms)
            if idle_disconnect_ms and ble_server._connections and time.ticks_diff(
                    time.ticks_ms(), last_activity) >= (idle_disconnect_ms if received_command else 10000):
                for connection in ble_server._connections.copy():
                    ble_server._ble.gap_disconnect(connection)
                last_activity = time.ticks_ms()
    finally:
        led.value(1)
        ble_server.stop()
        ble_server._advertise = original_advertise
        ble_server._ble_irq = original_irq


if __name__ == '__main__':
    main()
