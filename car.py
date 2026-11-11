"""Student API: BLE commands and camera feedback for a numbered robot."""
import math
import time
from gps import GPS
from mecanum_client import MecanumBLEClient


class Car:
    """Car(4) connects to BLE name car4 and reads ArUco marker 4.

    Construction waits for BLE, but never waits for camera visibility.
    Use a with block to stop/disconnect even when your script raises an error.
    """
    def __init__(self, number, *, timeout=15):
        if type(number) is not int or number < 0:
            raise ValueError('car number must be a nonnegative integer')
        self.number = number
        self.gps = GPS()
        self._pose_missing = None
        self._client = MecanumBLEClient(device_name=f'car{number}')
        try:
            self._client.connect()
            self._client.wait_until_connected(timeout)
        except BaseException:
            self._client.disconnect()
            raise

    @property
    def pose(self):
        """Pose(x, y, angle), or None when missing/stale. Angle is degrees CCW."""
        pose = self.gps.get_car(self.number)
        missing = pose is None
        if missing != self._pose_missing:
            if missing:
                problem = self.gps.feedback_problem or f'Marker {self.number} is not visible to the camera.'
                print(f'[car{self.number}] Position unavailable: {problem}', flush=True)
            else:
                print(f'[car{self.number}] Camera position available.', flush=True)
            self._pose_missing = missing
        return pose

    def wait_for_position(self, timeout=10):
        """Wait for a fresh pose, with diagnostics; timeout leaves cleanup to with."""
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be positive and finite')
        print(f'[car{self.number}] Waiting for camera position...', flush=True)
        deadline = time.monotonic() + timeout
        while True:
            pose = self.pose
            if pose is not None:
                return pose
            if time.monotonic() >= deadline:
                problem = self.gps.feedback_problem or f'Check marker {self.number} visibility.'
                raise TimeoutError(f'No position for car{self.number}. {problem}')
            time.sleep(min(.05, max(0, deadline - time.monotonic())))

    @property
    def position(self):
        pose = self.pose
        return None if pose is None else (pose.x, pose.y)

    @property
    def is_connected(self):
        return self._client.is_connected

    @property
    def connection_generation(self):
        return self._client.connection_generation

    def set_velocity(self, x=0., y=0., w=0.):
        """Normalized robot-frame forward, left, CCW commands, each in [-1, 1]."""
        values = dict(x=x, y=y, w=w)
        if not all(math.isfinite(v) and -1 <= v <= 1 for v in values.values()):
            raise ValueError('velocity must be finite and within [-1, 1]')
        if not self.is_connected:
            raise ConnectionError('Robot is disconnected')
        self._client.set_velocity(values)

    def go_to(self, x, y, angle):
        """Start onboard position control toward a field waypoint."""
        return self._client.set_waypoint(x, y, angle)

    def set_trim(self, matrix=None):
        """Replace the 3x3 correction matrix; no argument restores identity."""
        return self._client.set_trim(matrix)

    def read_trim(self):
        """Read the correction matrix currently stored on the robot."""
        return self._client.read_trim()

    @property
    def trim_matrix(self):
        """Read-back rows are output x/y/w; columns are requested x/y/w."""
        return self._client.trim_matrix

    def set_controller(self, parameters):
        """Validate, store, and read back onboard controller parameters."""
        return self._client.set_controller(parameters)

    def read_controller(self):
        """Read the controller parameters currently stored on the robot."""
        return self._client.read_controller()

    @property
    def controller_config(self):
        return self._client.controller_config

    def read_status(self):
        """Read current onboard mode, fault, pose, and telemetry counters."""
        return self._client.read_status()

    def reconnect_telemetry(self):
        """Stop and restart the robot's Wi-Fi/telemetry connection."""
        self._client.reconnect_telemetry()

    def restart(self, timeout=45):
        """Stop and reboot the robot, then wait for BLE to reconnect stopped."""
        self._client.restart_robot(timeout)

    def stop(self):
        self._client.stop()

    def close(self):
        self._client.disconnect()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
