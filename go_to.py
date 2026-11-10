"""Connect once, then accept field waypoints from the terminal."""
import argparse
import math
import time

from car import Car
from position_controller import PredictiveController, wrap_angle


POSITION_TOLERANCE = .025       # metres
HEADING_TOLERANCE = 5.          # degrees
HOLD_TIME = .6                  # seconds continuously inside tolerance
MIN_SPEED = .12                 # measured drivetrain dead zone
MIN_YAW = .08


def at_target(pose, target):
    position_error = math.hypot(target[0]-pose.x, target[1]-pose.y)
    heading_error = abs(math.degrees(wrap_angle(math.radians(target[2]-pose.angle))))
    return position_error <= POSITION_TOLERANCE, heading_error <= HEADING_TOLERANCE


def parse_waypoint(text):
    """Parse `x y w` or `x, y, w`; X/Y are metres and W is degrees."""
    parts=text.replace(',',' ').split()
    if len(parts) != 3:
        raise ValueError('enter three values: x y w')
    target=tuple(float(value) for value in parts)
    if not all(math.isfinite(value) for value in target):
        raise ValueError('waypoint values must be finite')
    if not -0.8 <= target[0] <= 0.8 or not -1.5 <= target[1] <= 1.5:
        raise ValueError('waypoint is outside the configured track bounds')
    return target


def move(car, target, timeout=30., controller=None):
    controller = controller or PredictiveController()
    deadline = time.monotonic()+timeout
    stable_since = None
    last_frame = None
    command = (0.,0.,0.)
    while time.monotonic() < deadline:
        pose = car.pose
        if pose is None:
            car.stop()
            raise RuntimeError('Camera telemetry was lost; the car was stopped.')
        position_ok, heading_ok = at_target(pose,target)
        if position_ok and heading_ok:
            command = (0.,0.,0.)
            stable_since = stable_since or time.monotonic()
            if time.monotonic()-stable_since >= HOLD_TIME:
                car.stop()
                return pose
        else:
            stable_since = None
            frame = car.gps.latest.received_at
            if frame != last_frame:
                output = controller.go_to(
                    pose.x,pose.y,math.radians(pose.angle),
                    tx=None if position_ok else target[0],
                    ty=None if position_ok else target[1],
                    ttheta=None if heading_ok else math.radians(target[2]))
                speed = math.hypot(output['x'],output['y'])
                if not position_ok and 0 < speed < MIN_SPEED:
                    scale = MIN_SPEED/speed
                    output['x'] *= scale
                    output['y'] *= scale
                if not heading_ok and 0 < abs(output['w']) < MIN_YAW:
                    output['w'] = math.copysign(MIN_YAW,output['w'])
                command = output['x'],output['y'],output['w']
                last_frame = frame
        car.set_velocity(*command)
        time.sleep(.05)
    car.stop()
    raise TimeoutError(f'Waypoint was not reached within {timeout:g} seconds.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--car',type=int,default=1,help='robot number (default: 1)')
    parser.add_argument('--timeout',type=float,default=30.)
    args=parser.parse_args(argv)
    if args.car < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('car must be nonnegative and timeout positive')
    with Car(args.car) as car:
        try:
            car.wait_for_position()
            controller=PredictiveController.for_car(args.car)
            print('Connected. Enter: x y w  (metres, metres, degrees CCW)')
            print('Type quit to stop and disconnect.')
            while True:
                try:
                    line=input('waypoint> ').strip()
                except EOFError:
                    break
                if line.lower() in ('q','quit','exit'):
                    break
                if not line:
                    continue
                try:
                    target=parse_waypoint(line)
                except ValueError as error:
                    print(f'Invalid waypoint: {error}')
                    continue
                print(f'Moving to ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.1f} deg)...')
                try:
                    pose=move(car,target,args.timeout,controller)
                    print(f'Reached: ({pose.x:.3f}, {pose.y:.3f}, {pose.angle:.1f} deg)')
                except (RuntimeError,TimeoutError) as error:
                    car.stop()
                    print(f'Stopped: {error}')
        except KeyboardInterrupt:
            print('\nStopped.')
        finally:
            car.stop()


if __name__=='__main__':
    main()
