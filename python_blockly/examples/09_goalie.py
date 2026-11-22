"""Follow the first ball's current X while facing 90 degrees at the bottom."""
from time import sleep
from vsss import Car, gps, run


@run
def main():
    car1 = Car(1)
    response = 2  # Try 8 for the hard scenario: same rule, quicker response.
    car1.wait_for_position()
    while True:
        balls = gps.balls
        if len(balls) > 0:
            ball = balls[0]
            car1.set_velocity(x=0, y=min(max((car1.pose.x - ball.x) * response, -1), 1), w=0)
        else:
            car1.stop()
        sleep(.05)


if __name__ == '__main__':
    main()
