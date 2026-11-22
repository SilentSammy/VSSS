"""One stationary ball and a clear approach; aim straight toward +Y."""
from time import monotonic, sleep
from vsss import Car, gps, run


@run
def main():
    car1 = Car(1)
    balls = gps.balls
    if len(balls) == 0:
        raise RuntimeError('One visible stationary ball is required.')
    ball = balls[0]
    car1.go_to(x=ball.x, y=ball.y - .3, angle=90)
    arrival_deadline = monotonic() + 10
    sleep(.1)
    while True:
        arrival_status = car1.read_status()
        if arrival_status.fault_name != 'none' or not arrival_status.pose_fresh:
            raise RuntimeError('Onboard position/control unavailable')
        if arrival_status.mode_name == 'reached':
            break
        if monotonic() >= arrival_deadline:
            raise TimeoutError('Waypoint timed out')
        sleep(.05)
    car1.set_velocity(x=1, y=0, w=0)
    sleep(.8)
    car1.stop()


if __name__ == '__main__':
    main()
