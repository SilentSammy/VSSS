"""Move car 1 between two field waypoints forever. Ctrl+C stops it.

Targets are (X metres, Y metres, heading degrees). Each target must be reached
before the next is sent. A fault, missing onboard pose, or timeout ends the run.
"""
from time import monotonic, sleep
from vsss import Car, run


@run
def main():
    car = Car(1)
    sleep(5)
    while True:
        for target in ((0, 0.5, 90), (0, -0.5, 90)):
            car.go_to(*target)
            deadline = monotonic() + 10
            sleep(0.1)
            while True:
                status = car.read_status()
                if status.fault_name != 'none' or not status.pose_fresh:
                    raise RuntimeError('Onboard position/control unavailable')
                if status.mode_name == 'reached':
                    break
                if monotonic() >= deadline:
                    raise TimeoutError('Waypoint timed out')
                sleep(0.05)


if __name__ == '__main__':
    main()
