"""Observe the field without connecting to any car over BLE. Ctrl+C exits."""
from time import sleep
from vsss import gps, run


@run
def main():
    while True:
        if gps.feedback_problem:
            print(gps.feedback_problem)
        else:
            print('Cars:', gps.cars)    # {id: Pose(x, y, angle)}
            print('Balls:', gps.balls)  # tuple of Position(x, y)
        # Missing/stale objects are omitted; an empty field can be valid.
        sleep(0.1)


if __name__ == '__main__':
    main()
