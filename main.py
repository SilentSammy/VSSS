"""A read-only starting point for a student robot program."""
from vsss import Car, GPS


def main():
    gps = GPS()
    try:
        with Car(1) as car:
            pose = car.wait_for_position()
            print("Car 1 pose:", pose)
            # Add movement commands here when the track is ready.
    finally:
        gps.close()


if __name__ == "__main__":
    main()
