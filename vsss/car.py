"""Student robot conveniences over one generic provider connection per car."""
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time
from .models import Pose, Velocity, finite
from .provider import VSSSProvider
from .gps import GPS


class Car:
    _cars = set()
    _cars_lock = Lock()

    def __init__(self, number, *, timeout=15, provider=None, endpoint=None):
        finite(timeout)
        if timeout <= 0: raise ValueError('timeout must be positive')
        if type(number) is not int or number < 0: raise ValueError('car number must be nonnegative')
        self.number = number
        self._provider = provider or VSSSProvider.connect(endpoint, timeout=timeout)
        self._owns_provider = provider is None
        self._closed, self._gps = False, None
        self._trim = self._controller = self._robot_config = None
        try:
            self._commands.connect(number)
        except BaseException:
            if self._owns_provider: self._provider.close()
            raise
        with self._cars_lock: self._cars.add(self)

    @property
    def _commands(self):
        if self._closed: raise RuntimeError('Car is closed')
        return self._provider.cars

    @property
    def gps(self):
        if self._closed: raise RuntimeError('Car is closed')
        if self._gps is None: self._gps = GPS(provider=self._provider)
        return self._gps

    @property
    def pose(self): return self.gps.get_car(self.number)

    @property
    def position(self):
        pose = self.pose
        return None if pose is None else (pose.x, pose.y)

    def wait_for_position(self, timeout=10):
        """Wait for a current visible camera pose; fail on timeout."""
        finite(timeout)
        if timeout <= 0: raise ValueError('timeout must be positive')
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pose = self.pose
            if pose is not None: return pose
            time.sleep(.02)
        raise TimeoutError(self.gps.feedback_problem or f'Marker {self.number} not visible')

    @property
    def is_connected(self):
        if self._closed: return False
        try: return self._commands.connection(self.number)['connected']
        except (ConnectionError, OSError): return False

    @property
    def connection_generation(self):
        return self._commands.connection(self.number)['generation']

    def set_velocity(self, x=0., y=0., w=0.):
        """Normalized robot-frame forward, left and counterclockwise velocity."""
        return self._commands.set_velocity(self.number, Velocity(x, y, w))

    def go_to(self, x, y, angle):
        """Start an onboard field waypoint, in metres and CCW degrees."""
        return self._commands.go_to(self.number, Pose(x, y, angle))

    def stop(self): return self._commands.stop(self.number)
    def read_status(self): return self._commands.status(self.number)
    def read_trim(self):
        self._trim = self._commands.read_trim(self.number)
        return self._trim
    def set_trim(self, matrix=None):
        self._trim = self._commands.set_trim(self.number, matrix)
        return self._trim
    @property
    def trim_matrix(self): return self._trim
    def read_controller(self):
        self._controller = self._commands.read_controller(self.number)
        return self._controller
    def set_controller(self, parameters):
        self._controller = self._commands.set_controller(self.number, parameters)
        return self._controller
    @property
    def controller_config(self): return self._controller
    def read_robot_config(self):
        self._robot_config = self._commands.read_robot_config(self.number)
        return self._robot_config
    def set_robot_config(self, configuration):
        self._robot_config = self._commands.set_robot_config(self.number, configuration)
        self.number = self._robot_config['car_id']
        return self._robot_config
    def reconnect_telemetry(self): return self._commands.reconnect_telemetry(self.number)
    def restart(self, timeout=45):
        self._commands.restart(self.number)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self._commands.connect(self.number)  # Reconnect stopped without releasing a sim gate.
                return self.read_status()
            except (ConnectionError, OSError, RuntimeError): time.sleep(.2)
        raise TimeoutError('Car did not reconnect stopped')

    def close(self):
        if self._closed: return
        try:
            self._commands.release(self.number)
        finally:
            try:
                if self._gps: self._gps.close()
                if self._owns_provider: self._provider.close()
            finally:
                self._closed = True
                with self._cars_lock: self._cars.discard(self)

    @classmethod
    def close_all(cls):
        with cls._cars_lock: cars = list(cls._cars)
        if cars:
            with ThreadPoolExecutor(max_workers=len(cars)) as pool:
                futures = [pool.submit(car.close) for car in cars]
                for future in futures: future.result()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
