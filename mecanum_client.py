"""Student-facing robot client. BLE runs in ble_worker.py on Windows or Linux."""
import atexit
import json
import math
from pathlib import Path
import subprocess
import sys
from threading import Condition, Lock, Thread
import time

from ble_protocol import CONTROLLER_KEYS, RobotStatus, normalize_controller

def to_byte(val):
    """Convert -1.0 to +1.0 range to 0-255 byte using two's complement"""
    signed = max(-128, min(127, int(val * 128.0)))
    return signed & 0xFF

def merge_proportional(cmd_primary, cmd_secondary):
    # Primary command overrides secondary based on its magnitude
    cmd_final = {}
    
    # Handle all axes from both commands
    all_axes = set(cmd_primary.keys()) | set(cmd_secondary.keys())
    
    for axis in all_axes:
        primary_input = cmd_primary.get(axis, 0.0)    # Default to 0 if missing
        secondary_input = cmd_secondary.get(axis, 0.0) # Default to 0 if missing
        
        if abs(primary_input) < 0.05:  # No primary input
            cmd_final[axis] = secondary_input
        else:
            # Primary input interpolates between secondary and desired value
            # abs(primary_input) determines how much override (0 to 1)
            # sign(primary_input) determines direction
            override_strength = abs(primary_input)
            desired_value = 1.0 if primary_input > 0 else -1.0
            cmd_final[axis] = (1 - override_strength) * secondary_input + override_strength * desired_value
    
    return cmd_final

def get_user_cmd(gamepad_index: int = 0):
    import user_input as inp

    keymappings = [
        ('w', 's', 'd', 'a', 'e', 'q', 'c'),  # Gamepad 0
        ('i', 'k', 'l', 'j', 'o', 'u', 'm'),  # Gamepad 1
        # Add more tuples for additional gamepads if needed
    ]

    fwd, bck, right, left, cw, ccw, boost_key = keymappings[gamepad_index] if gamepad_index < len(keymappings) else keymappings[0]
    s = str(gamepad_index)
    slow = 0.5
    fast = 1.0
    boost = inp.get_bipolar_ctrl(high_key=boost_key, high_game=f'RT{s}')
    scale = slow + (fast - slow) * boost
    return {
        'x':  inp.get_bipolar_ctrl(fwd,  bck,   f'LY{s}') * scale,
        'y': -inp.get_bipolar_ctrl(right, left,  f'LX{s}') * scale,  # positive y is left according to right-handed coordinate system
        'w': -inp.get_bipolar_ctrl(cw,    ccw,   f'RX{s}') * scale
    }

def get_manual_override(cmd):
    user_cmd = get_user_cmd()
    return merge_proportional(user_cmd, cmd)


def parse_waypoint_input(text):
    """Parse one waypoint-console line without touching the robot."""
    text = text.strip()
    if not text:
        return 'empty', None
    command = text.lower()
    if command in ('stop', 'manual', 'quit', 'exit', 'help'):
        return ('quit' if command == 'exit' else command), None
    parts = text.replace(',', ' ').split()
    if len(parts) != 3:
        raise ValueError('enter three values: x y angle')
    try:
        target = tuple(float(value) for value in parts)
    except ValueError as error:
        raise ValueError('x, y, and angle must be numbers') from error
    if not all(math.isfinite(value) for value in target):
        raise ValueError('x, y, and angle must be finite')
    return 'waypoint', target


def print_waypoint_help():
    print("\nWaypoint mode")
    print("  x y angle   Send a waypoint in metres, metres, degrees")
    print("  stop        Cancel motion")
    print("  manual      Return to WASD/gamepad control")
    print("  help        Show these commands")
    print("  quit        Stop and exit\n")


def run_waypoint_console(cars):
    """Read live waypoint commands while the BLE worker retains its link."""
    print_waypoint_help()
    while True:
        try:
            line = input("waypoint> ")
        except EOFError:
            line = 'quit'
        try:
            action, target = parse_waypoint_input(line)
            if action == 'empty':
                continue
            if action == 'help':
                print_waypoint_help()
                continue
            if action == 'manual':
                for car, _ in cars:
                    if car.is_connected:
                        car.stop()
                print("Manual mode. Press P to return to waypoint mode.")
                return 'manual'
            if action == 'quit':
                for car, _ in cars:
                    if car.is_connected:
                        car.stop()
                return 'quit'
            if action == 'stop':
                for car, _ in cars:
                    if car.is_connected:
                        car.stop()
                print("Stopped.")
                continue

            connected = [car for car, _ in cars if car.is_connected]
            if not connected:
                print("No car is connected yet.")
                continue
            for car in connected:
                car.set_waypoint(*target)
            print("Waypoint sent: x={:.3f}, y={:.3f}, angle={:.1f} deg".format(*target))
        except (ValueError, ConnectionError, RuntimeError, TimeoutError) as error:
            print(f"Invalid command: {error}")


def run_manual_demo(cars):
    """Run sampled keyboard/gamepad control until P switches modes."""
    import user_input as inp

    # Discard a P edge captured while the user was typing in the waypoint
    # console (for example, the final letter in "stop").
    inp.rising_edge('p')
    while True:
        for car, gamepad_index in cars:
            if car.is_connected:
                car.set_velocity(get_user_cmd(gamepad_index))
        if inp.rising_edge('p'):
            for car, _ in cars:
                if car.is_connected:
                    car.stop()
            return 'waypoint'
        time.sleep(0.02)

def normalize_trim_matrix(matrix=None):
    """Validate a matrix without connecting to hardware."""
    if matrix is None:
        matrix = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
    try:
        rows = tuple(tuple(float(v) for v in row) for row in matrix)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('trim must be a finite 3x3 matrix') from error
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError('trim must be a 3x3 matrix')
    values = tuple(v for row in rows for v in row)
    if not all(math.isfinite(v) and (0 if i in (0, 4, 8) else -2) <= v <= 2
               for i, v in enumerate(values)):
        raise ValueError('diagonal gains must be in [0, 2], cross terms in [-2, 2], all finite')
    return rows


class MecanumBLEClient:
    """Small synchronous interface to the separate BLE owner process."""
    def __init__(self, device_name="MP_BLE_Device", resolution=0.05):
        if not math.isfinite(resolution) or resolution <= 0:
            raise ValueError("resolution must be positive and finite")
        self.device_name = device_name
        self.resolution = resolution
        self.x = self.y = self.w = 0.0
        self.last_error = None
        self.trim_matrix = None
        self.controller_config = None
        self.robot_status = None
        self._operation_request = 0
        self._operation_results = {}
        self._waypoint_request = 0
        self._waypoint_results = {}
        self._process = None
        self._reader = None
        self._ready = False
        self._connection_generation = 0
        self._sequence = self._written = 0
        self._cached_velocity = None
        self._pipe_lock = Lock()
        self._status = Condition()
        atexit.register(self.disconnect)

    @property
    def is_connected(self):
        """True only after GATT discovery and the initial stop command finish."""
        return self._ready and self._process is not None and self._process.poll() is None

    @property
    def connection_generation(self):
        """Increase whenever the worker establishes a new ready BLE session."""
        return self._connection_generation

    def _read_status(self, process):
        try:
            for line in process.stdout:
                message = json.loads(line)
                with self._status:
                    if process is not self._process:
                        return
                    if 'ready' in message:
                        if message['ready'] and not self._ready:
                            self._connection_generation += 1
                        self._ready = message['ready']
                        self._cached_velocity = None
                    if 'written' in message:
                        self._written = max(self._written, message['written'])
                    if 'trim_matrix' in message:
                        values = message['trim_matrix']
                        self.trim_matrix = None if values is None else tuple(tuple(values[i:i+3]) for i in (0, 3, 6))
                    if 'controller_config' in message:
                        values = message['controller_config']
                        self.controller_config = (None if values is None else
                                                  {key: values[key] for key in CONTROLLER_KEYS})
                    if 'robot_status' in message:
                        self.robot_status = RobotStatus(*message['robot_status'])
                    if 'operation_done' in message:
                        self._operation_results[message['operation_done']] = message.get('operation_error')
                    if 'waypoint_done' in message:
                        self._waypoint_results[message['waypoint_done']] = message.get('waypoint_error')
                    if 'error' in message:
                        self.last_error = message['error']
                        print(f"[{self.device_name}] {self.last_error}")
                    if 'message' in message:
                        print(f"[{self.device_name}] {message['message']}")
                    self._status.notify_all()
        finally:
            with self._status:
                if process is self._process:
                    self._ready = False
                self._status.notify_all()

    def connect(self):
        """Start discovery/connection in the background; safe to call twice."""
        if self._process is not None:
            if self._process.poll() is None:
                return
            self.disconnect()
        self.last_error = None
        self._ready = False
        self._cached_velocity = None
        self._process = subprocess.Popen(
            [sys.executable, '-u', str(Path(__file__).with_name('ble_worker.py')), self.device_name],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding='utf-8', bufsize=1, close_fds=True)
        self._reader = Thread(target=self._read_status, args=(self._process,), daemon=True)
        self._reader.start()

    def wait_until_connected(self, timeout=15):
        """Optional blocking helper for scripts that need the robot immediately."""
        with self._status:
            ready = self._status.wait_for(lambda: self.is_connected or self._process is None
                                          or self._process.poll() is not None, timeout)
        if not ready or not self.is_connected:
            raise TimeoutError(self.last_error or f"Could not connect to {self.device_name}")

    def _send_velocity(self, velocity, force):
        with self._pipe_lock:
            if not self.is_connected or (not force and velocity == self._cached_velocity):
                return None
            self._sequence += 1
            message = dict(velocity=list(velocity), sequence=self._sequence, force=force)
            try:
                self._process.stdin.write(json.dumps(message) + '\n')
                self._process.stdin.flush()
            except (BrokenPipeError, OSError):
                self._ready = False
                return None
            self._cached_velocity = velocity
            return self._sequence

    def send(self, force=False):
        """Send the current x/y/w fields; the worker coalesces rapid updates."""
        self._send_velocity(bytes(to_byte(v) for v in (self.x, self.y, self.w)), force)

    def set_velocity(self, velocity, force=False):
        """Set normalized x/y/w, rounded to the configured resolution."""
        self.x, self.y, self.w = (round(velocity.get(axis, 0.0) / self.resolution) * self.resolution
                                 for axis in ('x', 'y', 'w'))
        self.send(force)

    def stop(self):
        """Send zero and wait briefly for its BLE acknowledgement."""
        self.x = self.y = self.w = 0.0
        sequence = self._send_velocity(bytes(3), force=True)
        if sequence is not None:
            with self._status:
                done = self._status.wait_for(lambda: self._written >= sequence
                                            or not self.is_connected, timeout=2)
            if not done:
                raise TimeoutError("Robot did not acknowledge stop")

    def _operation(self, operation, **values):
        with self._pipe_lock:
            if not self.is_connected:
                raise ConnectionError('Robot is disconnected')
            self._operation_request += 1
            request = self._operation_request
            message = dict(operation=operation, request=request, **values)
            self._process.stdin.write(json.dumps(message) + '\n')
            self._process.stdin.flush()
        with self._status:
            done = self._status.wait_for(lambda: request in self._operation_results
                                         or not self.is_connected, timeout=4)
            if not done or request not in self._operation_results:
                raise ConnectionError('BLE operation was not acknowledged; reconnect before retrying')
            error = self._operation_results.pop(request)
            if error:
                raise RuntimeError(error)

    def set_trim(self, matrix=None):
        """Write/read back a complete 3x3 matrix. Rows: outputs; columns: inputs.

        Diagonal entries are gains in [0, 2], cross terms in [-2, 2].
        Omitting the matrix resets it to identity. Settings last until reboot.
        """
        rows = normalize_trim_matrix(matrix)
        values = tuple(v for row in rows for v in row)
        self._operation('write_trim', values=values)
        return self.trim_matrix

    def read_trim(self):
        self._operation('read_trim')
        return self.trim_matrix

    def set_controller(self, parameters):
        configuration = normalize_controller(parameters, self.controller_config)
        self._operation('write_controller',
                        values=[configuration[key] for key in CONTROLLER_KEYS])
        return self.controller_config

    def read_controller(self):
        self._operation('read_controller')
        return self.controller_config

    def read_status(self):
        self._operation('read_status')
        return self.robot_status

    def reconnect_telemetry(self):
        """Stop motion and ask the robot to reconnect Wi-Fi and UDP telemetry."""
        self.stop()
        self._operation('reconnect_telemetry')

    def restart_robot(self, timeout=45):
        """Stop, request reboot, then wait for a fresh stopped connection."""
        self.stop()
        self._operation('restart')
        # Some Windows BLE stacks leave is_connected true after the ESP32
        # resets. Tear down our worker explicitly before starting a fresh one.
        self.disconnect()
        time.sleep(1)  # Firmware schedules reset 750 ms after acknowledgement.
        self.connect()
        self.wait_until_connected(timeout)

    def set_waypoint(self, x, y, angle):
        """Hand a field waypoint (metres, metres, degrees CCW) to onboard control."""
        target = tuple(float(value) for value in (x, y, angle))
        if (not all(math.isfinite(value) for value in target) or
                not -0.8 <= target[0] <= 0.8 or not -1.5 <= target[1] <= 1.5):
            raise ValueError('waypoint is invalid or outside the configured track')
        with self._pipe_lock:
            if not self.is_connected:
                raise ConnectionError('Robot is disconnected')
            self._waypoint_request += 1
            request = self._waypoint_request
            self._process.stdin.write(json.dumps(dict(waypoint=target, request=request)) + '\n')
            self._process.stdin.flush()
        with self._status:
            done = self._status.wait_for(lambda: request in self._waypoint_results
                                        or not self.is_connected, timeout=4)
            if not done or request not in self._waypoint_results:
                raise ConnectionError('Waypoint was not acknowledged; reconnect before retrying')
            error = self._waypoint_results.pop(request)
            if error:
                raise RuntimeError(error)
        return target

    def disconnect(self):
        """Close the pipe: the worker stops the robot and releases BLE, then exits."""
        process = self._process
        if process is None:
            return
        self._ready = False
        with self._pipe_lock:
            try:
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()  # Firmware timeout handles an unresponsive worker.
            process.wait()
        if self._reader is not None:
            self._reader.join(timeout=1)
        process.stdout.close()
        self._process = None
        self._cached_velocity = None

    def __enter__(self):
        self.connect()
        try:
            self.wait_until_connected()
        except Exception:
            self.disconnect()
            raise
        return self

    def __exit__(self, *exc):
        self.disconnect()

if __name__ == "__main__":
    import time

    print("Mecanum BLE Manual Control Demo")
    print("================================\n")

    # Each entry: (device_name, gamepad_index)
    # Add more tuples to control multiple cars simultaneously.
    CAR_CONFIG = [
        ("car1", 0),
        # ("car2", 0),
        # ("car3", 1),
    ]

    cars = [(MecanumBLEClient(device_name=name), gp_idx) for name, gp_idx in CAR_CONFIG]

    for car, _ in cars:
        car.connect()  # non-blocking; reconnect loop starts in background

    try:
        print(f"Controlling {len(cars)} car(s). Use WASD/gamepad to control.")
        print("  W/S or Left Stick Y: Forward/Backward (X-axis)")
        print("  A/D or Left Stick X: Strafe Left/Right (Y-axis)")
        print("  Q/E or Right Stick X: Rotate (W-axis)")
        print("  C: Full speed mode")
        print("  P: Switch to waypoint terminal")
        print("  Ctrl+C: Exit\n")

        mode = 'manual'
        while mode != 'quit':
            mode = run_manual_demo(cars) if mode == 'manual' else run_waypoint_console(cars)

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        for car, _ in cars:
            car.disconnect()  # The worker sends stop before disconnecting.
