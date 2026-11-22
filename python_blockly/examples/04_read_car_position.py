"""Read car 1's camera pose without opening BLE or commanding motion."""
from time import sleep
from vsss import gps, run


@run
def main():
    while True:
        pose = gps.get_car(1)
        print(pose)  # Pose(x, y, angle), or None if missing/stale.
        sleep(0.25)


if __name__ == '__main__':
    main()
