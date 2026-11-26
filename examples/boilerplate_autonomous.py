"""Use field feedback and onboard waypoint control together.

Running this example moves car 1 to TARGET. Use a clear track.
"""
from time import monotonic, sleep
from vsss import Car, gps, run


CAR_ID = 1
TARGET = (0.0, 0.75, 90.0)  # Field X/Y metres, heading degrees CCW.


@run
def main():
    car = Car(CAR_ID)
    car.wait_for_position(timeout=10)  # Desktop sees this car's marker.
    if not car.read_status().pose_fresh:
        raise RuntimeError('The car is not receiving its own position.')

    car.go_to(*TARGET)  # Accepted by firmware; movement continues onboard.
    deadline = monotonic() + 10
    while True:
        cars, balls = gps.cars, gps.balls
        if gps.feedback_problem or CAR_ID not in cars:
            raise RuntimeError('Field feedback lost.')

        # Add planning here using cars[CAR_ID], the other cars, and balls.
        # Check any other required car IDs or ball visibility explicitly.
        status = car.read_status()
        if status.fault or not status.pose_fresh:
            raise RuntimeError(f'Onboard control unavailable: {status.fault_name}')
        if status.mode_name == 'reached':
            print('Target reached.')
            break
        if monotonic() >= deadline:
            raise TimeoutError('Waypoint timed out.')
        sleep(0.1)
    # @run stops all cars and releases GPS, including on any error above.


if __name__ == '__main__':
    main()
