"""Track the first ball's X with waypoints along the top goal lane."""
from time import sleep
from vsss import Car, gps, run


@run
def main():
    car1 = Car(1)
    while True:
        balls = gps.balls
        if len(balls) > 0:
            ball = balls[0]
            car1.go_to(x=min(max(ball.x, -.65), .65), y=1.2, angle=0)
        else:
            car1.stop()
        sleep(.1)


if __name__ == '__main__':
    main()
