"""Student starting point: Car #1, shared positions, and a one-second spin."""
import time

from car import Car
from gps import GPS


def main():
    gps = GPS()  # Receives every visible car and ball from the lab desktop.
    try:
        with Car(1) as car:
            car.wait_for_position()

            # Replace this short loop with your own control logic.
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                pose = car.pose  # Metres and degrees; None if missing or stale.
                if pose is None:
                    break
                cars = gps.cars  # All visible cars, indexed by marker number.
                balls = gps.balls  # All detected ball positions.
                car.set_velocity(x=0, y=0, w=.25)
                time.sleep(.1)

            car.stop()
    finally:
        gps.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass  # The with block stops/disconnects the robot.
