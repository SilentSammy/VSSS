"""GPIO8 LED test using the car's existing BLE server; no motor imports."""
from machine import Pin
import network
import time
import ble_server

LED_UUID = '12345678-1234-5678-1234-56789abcdef1'


def main():
    # Keep the WiFi radio off when comparing BLE responsiveness.
    network.WLAN(network.STA_IF).active(False)
    network.WLAN(network.AP_IF).active(False)
    led = Pin(8, Pin.OUT, value=1)
    ble_server.DEVICE_NAME = 'VSSS_LED_TEST'

    def set_led(data):
        if len(data) == 1 and data[0] in (0, 1):
            led.value(1 - data[0])  # Onboard LED is active low.
            # Expose the applied state for the host's initial verification.
            ble_server._ble.gatts_write(ble_server._handles[LED_UUID], data)

    ble_server.control_callbacks = {LED_UUID: set_led}
    try:
        ble_server.start()
        print('GPIO8 BLE LED ready: VSSS_LED_TEST')
        while True:
            time.sleep_ms(100)
    finally:
        led.value(1)
        ble_server.stop()


if __name__ == '__main__':
    main()
