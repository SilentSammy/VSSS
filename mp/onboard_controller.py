"""Allocation-light predictive waypoint controller for MicroPython."""
import math
import time


CONTROLLER_KEYS = ("kp_dist", "kp_heading", "max_speed", "max_w", "horizon",
                   "velocity_alpha", "min_speed", "min_w", "position_tolerance",
                   "heading_tolerance_deg", "hold_ms", "position_hysteresis",
                   "heading_hysteresis", "telemetry_recovery_ms", "waypoint_timeout_ms")

DEFAULT_CONTROLLER = {
    "kp_dist": 4.0,
    "kp_heading": .55,
    "max_speed": .4,
    "max_w": .35,
    "horizon": .2,
    "velocity_alpha": .5,
    "min_speed": .12,
    "min_w": .08,
    "position_tolerance": .025,
    "heading_tolerance_deg": 5.0,
    "hold_ms": 600,
    "position_hysteresis": 1.5,
    "heading_hysteresis": 1.5,
    "telemetry_recovery_ms": 3000,
    "waypoint_timeout_ms": 30000,
}


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _ticks_diff(now, then):
    ticks_diff = getattr(time, "ticks_diff", None)
    return now - then if ticks_diff is None else ticks_diff(now, then)


class OnboardController:
    """Predictive field-to-robot waypoint controller."""

    def __init__(self, **overrides):
        parameters = dict(DEFAULT_CONTROLLER)
        unknown = set(overrides) - set(CONTROLLER_KEYS)
        if unknown:
            raise TypeError("unknown controller parameter: %s" % next(iter(unknown)))
        parameters.update(overrides)
        for key in CONTROLLER_KEYS:
            setattr(self, key, parameters[key])
        self.heading_tolerance = math.radians(self.heading_tolerance_deg)
        self.hold_ms = int(self.hold_ms)
        # At arrival, estimated travel over the hold period must fit inside
        # the tolerance. Derive limits without changing the BLE/config format.
        hold_s = max(.1, self.hold_ms / 1000.)
        self.settle_speed = self.position_tolerance / hold_s
        self.settle_yaw_rate = self.heading_tolerance / hold_s
        self.target = None
        self.reset()

    def reset(self):
        self.previous = None
        self.velocity = [0.0, 0.0, 0.0]
        self.command = (0.0, 0.0, 0.0)
        self.stable_since = None
        self.position_acquired = False
        self.heading_acquired = False

    def set_target(self, x, y, angle_radians):
        self.target = (float(x), float(y), float(angle_radians))
        self.reset()

    def clear_target(self):
        self.target = None
        self.reset()

    def update(self, pose, now_ms):
        """Consume `(x, y, heading_degrees)`; return robot-frame x/y/w."""
        if self.target is None or pose is None:
            self.stable_since = None
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
        if self.position_acquired:
            if position_error > self.position_tolerance * self.position_hysteresis:
                self.position_acquired = False
        elif position_error <= self.position_tolerance:
            self.position_acquired = True
        if self.heading_acquired:
            if heading_error > self.heading_tolerance * self.heading_hysteresis:
                self.heading_acquired = False
        elif heading_error <= self.heading_tolerance:
            self.heading_acquired = True
        position_ok = self.position_acquired
        heading_ok = self.heading_acquired
        translation_settled = math.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2) <= self.settle_speed
        rotation_settled = abs(self.velocity[2]) <= self.settle_yaw_rate
        if position_ok and heading_ok and translation_settled and rotation_settled:
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
        # While acquiring the arrival band, overcome loaded motor deadband.
        # Once acquired, latch and stop that axis until it leaves the wider
        # hysteresis band; this avoids both a near-target stall and chatter.
        if not position_ok and 0 < speed < self.min_speed:
            scale = self.min_speed / speed
            command_x *= scale
            command_y *= scale
        if not heading_ok and 0 < abs(command_w) < self.min_w:
            command_w = self.min_w if command_w > 0 else -self.min_w
        # Never fight a large heading correction with translation. Mecanum
        # cross-coupling can otherwise concentrate the combined command in one
        # wheel and create an alternating yaw/position limit cycle. Reacquire
        # heading in place, then resume translation with yaw latched at zero.
        if not heading_ok:
            command_x = command_y = 0.0
        elif position_ok:
            command_x = command_y = 0.0
        if heading_ok:
            command_w = 0.0
        self.command = (command_x, command_y, command_w)
        return self.command

    def reached(self, now_ms):
        return self.stable_since is not None and _ticks_diff(now_ms, self.stable_since) >= self.hold_ms

