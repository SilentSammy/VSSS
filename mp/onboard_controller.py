"""Allocation-light predictive waypoint controller for MicroPython."""
import math
import time


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _ticks_diff(now, then):
    ticks_diff = getattr(time, "ticks_diff", None)
    return now - then if ticks_diff is None else ticks_diff(now, then)


class OnboardController:
    """Same field-to-robot control law as the calibrated laptop controller."""

    def __init__(self, kp_dist=4.0, kp_heading=.55, max_speed=.4, max_w=.35,
                 horizon=.2, velocity_alpha=.5, min_speed=.12, min_w=.08,
                 position_tolerance=.025, heading_tolerance_deg=5., hold_ms=600):
        self.kp_dist = kp_dist
        self.kp_heading = kp_heading
        self.max_speed = max_speed
        self.max_w = max_w
        self.horizon = horizon
        self.velocity_alpha = velocity_alpha
        self.min_speed, self.min_w = min_speed, min_w
        self.position_tolerance = position_tolerance
        self.heading_tolerance = math.radians(heading_tolerance_deg)
        self.hold_ms = int(hold_ms)
        self.target = None
        self.reset()

    def reset(self):
        self.previous = None
        self.velocity = [0.0, 0.0, 0.0]
        self.command = (0.0, 0.0, 0.0)
        self.stable_since = None

    def set_target(self, x, y, angle_radians):
        self.target = (float(x), float(y), float(angle_radians))
        self.reset()

    def clear_target(self):
        self.target = None
        self.reset()

    def update(self, pose, now_ms):
        """Consume `(x, y, heading_degrees)`; return robot-frame x/y/w."""
        if self.target is None or pose is None:
            self.command = (0.0, 0.0, 0.0)
            return self.command
        x, y, angle_degrees = pose
        theta = math.radians(angle_degrees)
        if self.previous is not None:
            then, px, py, ptheta = self.previous
            dt = _ticks_diff(now_ms, then) / 1000.0
            if 0 < dt < 1.0:
                measured = ((x - px) / dt, (y - py) / dt,
                            wrap_angle(theta - ptheta) / dt)
                alpha = self.velocity_alpha
                for index in range(3):
                    self.velocity[index] = (alpha * measured[index] +
                                            (1 - alpha) * self.velocity[index])
        self.previous = (now_ms, x, y, theta)
        predicted_x = x + self.velocity[0] * self.horizon
        predicted_y = y + self.velocity[1] * self.horizon
        predicted_theta = theta + self.velocity[2] * self.horizon
        tx, ty, target_theta = self.target
        position_error = math.sqrt((tx - x) ** 2 + (ty - y) ** 2)
        heading_error = abs(wrap_angle(target_theta - theta))
        position_ok = position_error <= self.position_tolerance
        heading_ok = heading_error <= self.heading_tolerance
        if position_ok and heading_ok:
            self.stable_since = now_ms if self.stable_since is None else self.stable_since
            self.command = (0.0, 0.0, 0.0)
            return self.command
        self.stable_since = None
        board_x = self.kp_dist * (tx - predicted_x)
        board_y = self.kp_dist * (ty - predicted_y)
        magnitude = math.sqrt(board_x * board_x + board_y * board_y)
        if magnitude > self.max_speed:
            scale = self.max_speed / magnitude
            board_x *= scale
            board_y *= scale
        command_x = board_x * math.cos(theta) + board_y * math.sin(theta)
        command_y = -board_x * math.sin(theta) + board_y * math.cos(theta)
        command_w = self.kp_heading * wrap_angle(target_theta - predicted_theta)
        command_w = max(-self.max_w, min(self.max_w, command_w))
        speed = math.sqrt(command_x * command_x + command_y * command_y)
        # Do not impose a discontinuous minimum kick immediately outside the
        # arrival band.  Camera quantization would otherwise create a limit
        # cycle across the tolerance boundary.
        if (position_error >= 2 * self.position_tolerance and
                0 < speed < self.min_speed):
            scale = self.min_speed / speed
            command_x *= scale
            command_y *= scale
        if (heading_error >= 2 * self.heading_tolerance and
                0 < abs(command_w) < self.min_w):
            command_w = self.min_w if command_w > 0 else -self.min_w
        if position_ok:
            command_x = command_y = 0.0
        if heading_ok:
            command_w = 0.0
        self.command = (command_x, command_y, command_w)
        return self.command

    def reached(self, now_ms):
        return self.stable_since is not None and _ticks_diff(now_ms, self.stable_since) >= self.hold_ms

