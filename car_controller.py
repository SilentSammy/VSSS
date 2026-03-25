import json
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from simple_pid import PID


@dataclass
class Car:
    name: str
    aruco_id: int
    controller: 'CarController' = field(default_factory=lambda: CarController())


def load_cars(path='cars.json'):
    with open(path) as f:
        return [Car(**entry) for entry in json.load(f)]


class CarController:
    """Generates velocity commands to drive a mecanum car to a desired pose.

    Wraps PIDs for position and heading. Pass current state each call;
    the controller is stateless between calls except for PID internals.
    """

    def __init__(self,
                 kp_dist=5.0, kd_dist=0.2, max_speed=0.4,
                 kp_heading=0.8, kd_heading=0.1, max_w=0.7):
        self._pid_distance = PID(Kp=kp_dist, Ki=0, Kd=kd_dist, setpoint=0)
        self._pid_distance.output_limits = (-max_speed, max_speed)

        self._pid_heading = PID(Kp=kp_heading, Ki=0, Kd=kd_heading, setpoint=0)
        self._pid_heading.output_limits = (-max_w, max_w)

    def _board_to_robot(self, x_board, y_board, theta):
        """Rotate a board-frame vector into the robot frame."""
        x = x_board * np.cos(-theta) - y_board * np.sin(-theta)
        y = x_board * np.sin(-theta) + y_board * np.cos(-theta)
        return x, y

    def go_to(self, x, y, theta, tx=None, ty=None, ttheta=None, path_idx=None):
        """Return a velocity command dict to reach the desired pose.

        Args:
            x, y, theta: Current position (m) and heading (rad).
            tx, ty: Target position. None = don't care.
            ttheta: Target heading (rad). None = don't care.

        Returns:
            dict {'x', 'y', 'w'} — robot-frame velocity commands.
        """
        cmd = {'x': 0.0, 'y': 0.0, 'w': 0.0}

        if tx is not None or ty is not None:
            tx = tx if tx is not None else x
            ty = ty if ty is not None else y
            dx = tx - x
            dy = ty - y
            distance = np.sqrt(dx**2 + dy**2)

            speed = -self._pid_distance(distance)
            angle_to_target = np.arctan2(dy, dx)
            x_board = speed * np.cos(angle_to_target)
            y_board = speed * np.sin(angle_to_target)
            cmd['x'], cmd['y'] = self._board_to_robot(x_board, y_board, theta)

        if ttheta is not None:
            heading_error = -((ttheta - theta + np.pi) % (2 * np.pi) - np.pi)
            cmd['w'] = self._pid_heading(heading_error)

        return cmd

    def reset(self):
        self._pid_distance.reset()
        self._pid_heading.reset()


class Path:
    """A sequence of (x, y) points describing a trajectory."""

    def __init__(self, points, loop=True):
        self.pts = np.array(points, dtype=float)  # (N, 2)
        self.loop = loop

    @classmethod
    def circle(cls, r=0.15, n=200, cx=0.0, cy=0.0):
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return cls(np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)]))

    @classmethod
    def from_fn(cls, fn, t_start=0.0, t_end=2*np.pi, n=200, loop=True):
        """fn(t) -> (x, y), sampled at n evenly-spaced t values."""
        ts = np.linspace(t_start, t_end, n, endpoint=not loop)
        return cls([fn(t) for t in ts], loop=loop)

    def resample(self, spacing=0.005):
        """Return a new Path with points redistributed at uniform arc-length intervals."""
        pts = self.pts
        diffs = np.diff(pts, axis=0)
        seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
        cumlen = np.concatenate([[0], np.cumsum(seg_lens)])
        total = cumlen[-1]
        n = max(2, int(total / spacing))
        new_s = np.linspace(0, total, n, endpoint=not self.loop)
        new_x = np.interp(new_s, cumlen, pts[:, 0])
        new_y = np.interp(new_s, cumlen, pts[:, 1])
        return Path(np.column_stack([new_x, new_y]), loop=self.loop)

    def ordered_from(self, idx):
        """Points starting from idx; closes the loop by appending pts[idx] at end."""
        n = len(self.pts)
        count = n + 1 if self.loop else n - idx
        return [(float(self.pts[(idx + i) % n, 0]),
                 float(self.pts[(idx + i) % n, 1]))
                for i in range(count)]

    def roll_to_closest(self, x, y):
        """Return a new Path whose points start at the point nearest to (x, y)."""
        dists = np.hypot(self.pts[:, 0] - x, self.pts[:, 1] - y)
        idx = int(np.argmin(dists))
        return Path(np.roll(self.pts, -idx, axis=0), loop=self.loop)

    def reverse(self):
        """Return a new Path with traversal direction reversed, keeping the same start point."""
        pts = self.pts
        reversed_pts = np.concatenate([pts[:1], pts[1:][::-1]])
        return Path(reversed_pts, loop=self.loop)

    def __add__(self, other):
        """Concatenate two paths into one loop: self then other, back to self.pts[0]."""
        return Path(np.concatenate([self.pts, other.pts]), loop=True)


class SelfTuningCarController(CarController):
    """CarController that continuously adjusts its own distance PID gains online.

    Monitors a sliding window of distance errors each frame. When oscillation is
    detected it backs off Kp and boosts Kd; when the error trend is worsening it
    reduces Kp; when the error is steadily improving it nudges Kp up.  All gains
    are clamped to safe bounds so the system can't diverge.

    Usage:
        ctrl = SelfTuningCarController()
        cmd  = ctrl.go_to(x, y, theta, tx=tx, ty=ty, ttheta=ttheta)
        # No extra arguments needed — it monitors error automatically.
    """

    def __init__(self, window=80, lr=0.015,
                 kp_dist_bounds=(1.0, 15.0), kd_dist_bounds=(0.0, 1.5),
                 **kwargs):
        super().__init__(**kwargs)
        self._window = window
        self._lr = lr
        self._kp_bounds = kp_dist_bounds
        self._kd_bounds = kd_dist_bounds
        self._dist_errors = deque(maxlen=window)
        self.tuning_enabled = True  # set False when disconnected to avoid corrupting the error buffer

    def go_to(self, x, y, theta, tx=None, ty=None, ttheta=None, path_idx=None):
        if self.tuning_enabled and tx is not None and ty is not None:
            self._dist_errors.append(np.hypot(tx - x, ty - y))
            if len(self._dist_errors) == self._window:
                self._adapt()

        return super().go_to(x, y, theta, tx=tx, ty=ty, ttheta=ttheta, path_idx=path_idx)

    def _adapt(self):
        errors = np.array(self._dist_errors)

        # Oscillation: fraction of steps where the error derivative changes sign
        deriv = np.diff(errors)
        osc = np.mean(np.diff(np.sign(deriv)) != 0)

        # Trend: is the recent half worse than the older half?
        half = len(errors) // 2
        trend = np.mean(errors[half:]) - np.mean(errors[:half])

        kp = self._pid_distance.Kp
        kd = self._pid_distance.Kd

        if osc > 0.35:
            # Too oscillatory — soften Kp, harden Kd
            kp *= (1 - self._lr)
            kd *= (1 + self._lr * 0.5)
        elif trend > 0.005:
            # Error is growing — reduce Kp more aggressively
            kp *= (1 - self._lr * 1.5)
        elif trend < -0.002:
            # Error is shrinking — cautiously raise Kp
            kp *= (1 + self._lr * 0.3)

        self._pid_distance.Kp = float(np.clip(kp, *self._kp_bounds))
        self._pid_distance.Kd = float(np.clip(kd, *self._kd_bounds))


class PurePursuit:
    """Finds the lookahead target point on a Path."""

    def __init__(self, path, lookahead=0.05):
        self._path = path
        self.lookahead = lookahead
        self._idx = 0

    def get_target(self, x, y):
        """Advance along path and return the lookahead (tx, ty)."""
        pts = self._path.pts
        loop = self._path.loop
        n = len(pts)

        # Advance _idx to the closest point in a forward search window
        window = min(n, max(10, n // 5))
        best_d = np.hypot(pts[self._idx, 0] - x, pts[self._idx, 1] - y)
        for i in range(1, window):
            idx = (self._idx + i) % n if loop else min(self._idx + i, n - 1)
            d = np.hypot(pts[idx, 0] - x, pts[idx, 1] - y)
            if d < best_d:
                best_d = d
                self._idx = idx

        # Walk forward accumulating arc length until >= lookahead
        arc = 0.0
        prev = self._idx
        for _ in range(n):
            nxt = (prev + 1) % n if loop else min(prev + 1, n - 1)
            arc += np.hypot(pts[nxt, 0] - pts[prev, 0], pts[nxt, 1] - pts[prev, 1])
            if arc >= self.lookahead or (not loop and nxt == n - 1):
                return float(pts[nxt, 0]), float(pts[nxt, 1])
            prev = nxt

        return float(pts[(self._idx + 1) % n, 0]), float(pts[(self._idx + 1) % n, 1])

    def ordered_points(self):
        """Remaining path points from current progress, for visualization."""
        return self._path.ordered_from(self._idx)
