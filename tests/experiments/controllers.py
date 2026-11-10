"""Pose PID from main's car_controller.py, without config files or camera imports.

Positions are metres. Controller angles are radians; GPS pose.angle is degrees.
"""
import math
from simple_pid import PID
from position_controller import PredictiveController, wrap_angle


class CarController:
    def __init__(self, kp_dist=5., ki_dist=0., kd_dist=.2, max_speed=.4,
                 kp_heading=.8, ki_heading=0., kd_heading=.1, max_w=.5,
                 time_fn=None):
        self.distance = PID(kp_dist, ki_dist, kd_dist, setpoint=0,
                            output_limits=(-max_speed, max_speed),
                            sample_time=None, time_fn=time_fn)
        self.heading = PID(kp_heading, ki_heading, kd_heading, setpoint=0,
                           output_limits=(-max_w, max_w),
                           sample_time=None, time_fn=time_fn)

    def go_to(self, x, y, theta, tx=None, ty=None, ttheta=None):
        command = dict(x=0., y=0., w=0.)
        if tx is not None or ty is not None:
            dx = (x if tx is None else tx) - x
            dy = (y if ty is None else ty) - y
            distance = math.hypot(dx, dy)
            speed = -self.distance(distance)
            if distance > 0:
                # Board-frame target vector rotated by -theta into robot axes.
                command['x'] = speed * (dx * math.cos(theta) + dy * math.sin(theta)) / distance
                command['y'] = speed * (-dx * math.sin(theta) + dy * math.cos(theta)) / distance
        if ttheta is not None:
            command['w'] = self.heading(-wrap_angle(ttheta - theta))
        return command

    def reset(self):
        self.distance.reset()
        self.heading.reset()


def make_controller(args, clock):
    """P is a comparison baseline; PID is the original pose-controller logic.

    Experiment also accepts any controller implementing reset() and go_to().
    Predictive is the measured default; P and PID remain comparison baselines.
    """
    if args.controller == 'predictive':
        return PredictiveController(
            kp_dist=args.kp_dist, kp_heading=args.kp_heading,
            max_speed=args.center_speed, max_w=args.center_w,
            horizon=args.prediction_horizon,
            velocity_alpha=args.velocity_alpha, time_fn=clock)
    pid = args.controller == 'pid'
    return CarController(
        kp_dist=args.kp_dist, ki_dist=args.ki_dist if pid else 0., kd_dist=args.kd_dist if pid else 0.,
        kp_heading=args.kp_heading, ki_heading=args.ki_heading if pid else 0., kd_heading=args.kd_heading if pid else 0.,
        max_speed=args.center_speed, max_w=args.center_w, time_fn=clock)
