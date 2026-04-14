import time
import user_input as inp
from mecanum_client import MecanumBLEClient

DEVICE_NAME = "Therian00"

car = MecanumBLEClient(device_name=DEVICE_NAME)
car.connect()

try:
    while True:
        x = 0.0
        y = 0.0
        w = 0.0

        if inp.is_pressed('w'):
            x = 0.5
        if inp.is_pressed('s'):
            x = -0.5
        
        if inp.is_pressed('d'):
            y = -0.5
        if inp.is_pressed('a'):
            y = 0.5
        
        if inp.is_pressed('e'):
            w = -0.5
        if inp.is_pressed('q'):
            w = 0.5
        
        velocity = {'x': x, 'y': y, 'w': w}
        car.set_velocity(velocity)
        time.sleep(0.02)
finally:
    car.stop()
    car.disconnect()
