"""Small latency-aware controller for field-position waypoints."""
import json
import math
from pathlib import Path
import time


def wrap_angle(angle):
    """Wrap a radian angle to [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


class PredictiveController:
    """Predict current motion forward before applying bounded P control."""
    def __init__(self, kp_dist=4., kp_heading=.55, max_speed=.4, max_w=.35,
                 horizon=.2, velocity_alpha=.5, time_fn=time.monotonic):
        self.kp_dist, self.kp_heading = kp_dist, kp_heading
        self.max_speed, self.max_w = max_speed, max_w
        self.horizon, self.velocity_alpha = horizon, velocity_alpha
        self.clock = time_fn
        self.reset()

    @classmethod
    def for_car(cls, number, **overrides):
        """Load saved parameters when available, then apply explicit overrides."""
        path=Path(__file__).with_name('calibration')/f'controller_car{number}.json'
        parameters={}
        try:
            saved=json.loads(path.read_text(encoding='utf-8'))['parameters']
            names=dict(kp_distance='kp_dist',kp_heading='kp_heading',max_speed='max_speed',
                       max_yaw='max_w',prediction_horizon_s='horizon',
                       velocity_filter_alpha='velocity_alpha')
            parameters={target:saved[source] for source,target in names.items() if source in saved}
        except FileNotFoundError:
            pass
        parameters.update(overrides)
        return cls(**parameters)

    def reset(self):
        self.previous = None
        self.velocity = [0., 0., 0.]

    def go_to(self, x, y, theta, tx=None, ty=None, ttheta=None):
        now = self.clock()
        if self.previous is not None:
            then, px, py, ptheta = self.previous
            dt = now-then
            if 0 < dt < 1.:
                measured = ((x-px)/dt, (y-py)/dt, wrap_angle(theta-ptheta)/dt)
                alpha = self.velocity_alpha
                self.velocity = [alpha*new+(1-alpha)*old
                                 for new,old in zip(measured,self.velocity)]
        self.previous = (now,x,y,theta)
        predicted = (x+self.velocity[0]*self.horizon,
                     y+self.velocity[1]*self.horizon,
                     theta+self.velocity[2]*self.horizon)
        dx = 0. if tx is None else tx-predicted[0]
        dy = 0. if ty is None else ty-predicted[1]
        board_x, board_y = self.kp_dist*dx, self.kp_dist*dy
        magnitude = math.hypot(board_x,board_y)
        if magnitude > self.max_speed:
            board_x *= self.max_speed/magnitude
            board_y *= self.max_speed/magnitude
        command = dict(
            x=board_x*math.cos(theta)+board_y*math.sin(theta),
            y=-board_x*math.sin(theta)+board_y*math.cos(theta),
            w=0.)
        if ttheta is not None:
            command['w'] = max(-self.max_w,min(self.max_w,
                self.kp_heading*wrap_angle(ttheta-predicted[2])))
        return command
