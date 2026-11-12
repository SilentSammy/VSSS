"""Send robot-relative velocity commands over BLE; no GPS required.

Running this example moves car 1 forward briefly. Use a clear track.
"""
from time import sleep
from vsss import Car, run


@run
def main():
    car = Car(1)
    # Normalized forward, left, and counterclockwise rotation, each -1 to 1.
    car.set_velocity(x=0.2, y=0, w=0)
    sleep(1)
    car.stop()
    # @run also stops/disconnects every car on return, error, or Ctrl+C.


if __name__ == '__main__':
    main()
