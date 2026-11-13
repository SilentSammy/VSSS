"""Rotate car 1 CCW, then CW, forever. Ctrl+C stops and disconnects."""
from time import sleep
from vsss import Car, run


@run
def main():
    car = Car(1)
    while True:
        car.set_velocity(x=0, y=0, w=0.2)
        sleep(1)
        car.set_velocity(x=0, y=0, w=-0.2)
        sleep(1)


if __name__ == '__main__':
    main()
