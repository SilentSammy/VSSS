"""One stationary ball and a clear approach; aim straight toward +Y."""
from time import sleep
from vsss import Car, gps, run


@run
def main():
    car1 = Car(1)
    ball = gps.balls[0]
    car1.go_to(x=ball.x, y=ball.y - .3, angle=90)
    sleep(.1)
    while car1.read_status().mode_name != 'reached':
        sleep(.05)
    car1.set_velocity(x=1, y=0, w=0)
    sleep(.8)
    car1.stop()


if __name__ == '__main__':
    main()
