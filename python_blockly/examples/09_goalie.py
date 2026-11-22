"""Follow ball X using waypoints with fixed Y and heading."""
from time import sleep
from vsss import Car, gps, run


@run
def main():
    car1 = Car(1)
    sleep(5)
    while True:
        ball = gps.ball
        car1.go_to(x=ball.x, y=1.2, angle=-90)
        sleep(.05)


if __name__ == '__main__':
    main()
