"""Mecanum BLE control: three velocity bytes also act as heartbeats."""
import time
import ble_server
from mecanum import MecanumCar

DEVICE_NAME = "Eq4"
VELOCITY_UUID = '12345678-1234-5678-1234-56789abcdef1'


def main():
    car = MecanumCar()
    ble_server.DEVICE_NAME = DEVICE_NAME
    ble_server.disconnect_callback = car.stop

    def on_velocity(data):
        if len(data) == 3:
            car.set_velocity(*(ble_server.to_bipolar(value) for value in data))

    ble_server.control_callbacks = {VELOCITY_UUID: on_velocity}
    try:
        ble_server.start()
        while True:
            ble_server.poll()
            time.sleep_ms(20)
    except KeyboardInterrupt:
        pass
    finally:
        car.stop()
        ble_server.stop()


if __name__ == '__main__':
    main()
