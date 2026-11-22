"""Print gamepad 0 inputs. No car or GPS is used. Ctrl+C exits."""
from time import sleep
from vsss import input, run


@run
def main():
    while True:
        print('Buttons A/B/X/Y:', input.is_pressed('A0'), input.is_pressed('B0'),
              input.is_pressed('X0'), input.is_pressed('Y0'))
        print('Sticks LX/LY/RX/RY:', input.get_axis('LX0'), input.get_axis('LY0'),
              input.get_axis('RX0'), input.get_axis('RY0'))
        print('Triggers LT/RT:', input.get_axis('LT0'), input.get_axis('RT0'))
        sleep(0.25)


if __name__ == '__main__':
    main()
