import json
import numpy as np
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

    def go_to(self, x, y, theta, tx=None, ty=None, ttheta=None):
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


class PurePursuit:
    """Finds the lookahead target point on a discrete path."""

    def __init__(self, points, lookahead=0.05, loop=True):
        self._pts = np.array(points, dtype=float)  # (N, 2)
        self.lookahead = lookahead
        self.loop = loop
        self._idx = 0

    def get_target(self, x, y):
        """Advance along path and return the lookahead (tx, ty)."""
        pts = self._pts
        n = len(pts)

        # Advance _idx to the closest point in a forward search window
        window = min(n, max(10, n // 5))
        best_d = np.hypot(pts[self._idx, 0] - x, pts[self._idx, 1] - y)
        for i in range(1, window):
            idx = (self._idx + i) % n if self.loop else min(self._idx + i, n - 1)
            d = np.hypot(pts[idx, 0] - x, pts[idx, 1] - y)
            if d < best_d:
                best_d = d
                self._idx = idx

        # Walk forward accumulating arc length until >= lookahead
        arc = 0.0
        prev = self._idx
        for _ in range(n):
            nxt = (prev + 1) % n if self.loop else min(prev + 1, n - 1)
            arc += np.hypot(pts[nxt, 0] - pts[prev, 0], pts[nxt, 1] - pts[prev, 1])
            if arc >= self.lookahead or (not self.loop and nxt == n - 1):
                return float(pts[nxt, 0]), float(pts[nxt, 1])
            prev = nxt

        return float(pts[(self._idx + 1) % n, 0]), float(pts[(self._idx + 1) % n, 1])

    def ordered_points(self):
        """Path points reordered from current progress index; closes the loop if loop=True."""
        n = len(self._pts)
        count = n + 1 if self.loop else n
        return [(float(self._pts[(self._idx + i) % n, 0]),
                 float(self._pts[(self._idx + i) % n, 1]))
                for i in range(count)]
