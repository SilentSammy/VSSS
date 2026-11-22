"""Partial teleop: W/S drive at +/-50%; gamepad 0 RX turns at +/-50%.

Sideways motion is left unmapped for students to complete. Ctrl+C stops the car.
"""
from time import sleep
from vsss import Car, input, run


@run
def main():
    car = Car(1)
    while True:
        forward = input.get_bipolar_ctrl(high_key='w', low_key='s', keyboard_scale=0.5)
        turn = input.get_axis('RX0') * 0.5
        car.set_velocity(x=forward, y=0, w=turn)  # TODO: map a sideways input to y.
        sleep(0.05)


if __name__ == '__main__':
    main()
