import json
import numpy as np
import os
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
                 kp_dist=5.0, ki_dist=0.0, kd_dist=0.2, max_speed=0.4, ki_dist_limit=None,
                 kp_heading=0.8, ki_heading=0.0, kd_heading=0.1, max_w=0.7, ki_heading_limit=None,
                 tune_file='car_pid.json'):
        self._pid_distance = PID(Kp=kp_dist, Ki=ki_dist, Kd=kd_dist, setpoint=0)
        self._pid_distance.output_limits = (-max_speed, max_speed)
        if ki_dist_limit is not None:
            self._pid_distance.integral_limits = (-ki_dist_limit, ki_dist_limit)

        self._pid_heading = PID(Kp=kp_heading, Ki=ki_heading, Kd=kd_heading, setpoint=0)
        self._pid_heading.output_limits = (-max_w, max_w)
        if ki_heading_limit is not None:
            self._pid_heading.integral_limits = (-ki_heading_limit, ki_heading_limit)
        
        # Real-time tuning support
        self._tune_file = tune_file
        self._last_tune_mtime = 0.0
        self._create_tune_file_if_missing()

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
    
    def _create_tune_file_if_missing(self):
        """Create tune file with current parameters if it doesn't exist."""
        if not os.path.exists(self._tune_file):
            params = {
                "kp_dist": self._pid_distance.Kp,
                "ki_dist": self._pid_distance.Ki,
                "kd_dist": self._pid_distance.Kd,
                "max_speed": self._pid_distance.output_limits[1],
                "ki_dist_limit": self._pid_distance.integral_limits[1] if hasattr(self._pid_distance, 'integral_limits') and self._pid_distance.integral_limits else None,
                "kp_heading": self._pid_heading.Kp,
                "ki_heading": self._pid_heading.Ki,
                "kd_heading": self._pid_heading.Kd,
                "max_w": self._pid_heading.output_limits[1],
                "ki_heading_limit": self._pid_heading.integral_limits[1] if hasattr(self._pid_heading, 'integral_limits') and self._pid_heading.integral_limits else None
            }
            with open(self._tune_file, 'w') as f:
                json.dump(params, f, indent=2)
    
    def check_tune_file(self):
        """Check for tune file changes and reload parameters if modified.
        
        Call this once per loop iteration to enable real-time PID tuning.
        Returns True if parameters were reloaded, False otherwise.
        """
        if not os.path.exists(self._tune_file):
            return False
        
        try:
            mtime = os.path.getmtime(self._tune_file)
            if mtime != self._last_tune_mtime:
                self._last_tune_mtime = mtime
                with open(self._tune_file) as f:
                    params = json.load(f)
                
                # Apply distance PID params
                if 'kp_dist' in params:
                    self._pid_distance.Kp = params['kp_dist']
                if 'ki_dist' in params:
                    self._pid_distance.Ki = params['ki_dist']
                if 'kd_dist' in params:
                    self._pid_distance.Kd = params['kd_dist']
                if 'max_speed' in params:
                    self._pid_distance.output_limits = (-params['max_speed'], params['max_speed'])
                if 'ki_dist_limit' in params:
                    if params['ki_dist_limit'] is not None:
                        self._pid_distance.integral_limits = (-params['ki_dist_limit'], params['ki_dist_limit'])
                    else:
                        self._pid_distance.integral_limits = (None, None)
                
                # Apply heading PID params
                if 'kp_heading' in params:
                    self._pid_heading.Kp = params['kp_heading']
                if 'ki_heading' in params:
                    self._pid_heading.Ki = params['ki_heading']
                if 'kd_heading' in params:
                    self._pid_heading.Kd = params['kd_heading']
                if 'max_w' in params:
                    self._pid_heading.output_limits = (-params['max_w'], params['max_w'])
                if 'ki_heading_limit' in params:
                    if params['ki_heading_limit'] is not None:
                        self._pid_heading.integral_limits = (-params['ki_heading_limit'], params['ki_heading_limit'])
                    else:
                        self._pid_heading.integral_limits = (None, None)
                
                print(f"[PID] Reloaded: kp_dist={self._pid_distance.Kp:.2f} ki_dist={self._pid_distance.Ki:.3f} kd_dist={self._pid_distance.Kd:.2f} "
                      f"max_speed={self._pid_distance.output_limits[1]:.2f} | "
                      f"kp_heading={self._pid_heading.Kp:.2f} ki_heading={self._pid_heading.Ki:.3f} kd_heading={self._pid_heading.Kd:.2f} "
                      f"max_w={self._pid_heading.output_limits[1]:.2f}")
                return True
        except (OSError, ValueError, KeyError) as e:
            print(f"[PID] Failed to reload tune file: {e}")
        
        return False


class PurePursuit:
    """Finds the lookahead target point on an Nx2 array of (x, y) points."""

    def __init__(self, points, lookahead=0.05, loop=True):
        self.pts = np.asarray(points, dtype=float)  # (N, 2)
        self.lookahead = lookahead
        self.loop = loop
        self._idx = 0

    def get_target(self, x, y):
        """Advance along path and return the lookahead (tx, ty)."""
        pts = self.pts
        n = len(pts)
        loop = self.loop

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
        pts = self.pts
        n = len(pts)
        count = n + 1 if self.loop else n - self._idx
        return [(float(pts[(self._idx + i) % n, 0]),
                 float(pts[(self._idx + i) % n, 1]))
                for i in range(count)]
