"""Simple held-key control: W/S forward, A/D sideways, Q/E turning."""
from time import sleep
from vsss import Car, input, run


@run
def main():
    car1 = Car(1)
    while True:
        forward = 0
        left = 0
        turn = 0
        if input.is_pressed('w'):
            forward = .3
        if input.is_pressed('s'):
            forward = -.3
        if input.is_pressed('a'):
            left = .3
        if input.is_pressed('d'):
            left = -.3
        if input.is_pressed('q'):
            turn = .3
        if input.is_pressed('e'):
            turn = -.3
        car1.set_velocity(x=forward, y=left, w=turn)
        sleep(.05)


if __name__ == '__main__':
    main()
