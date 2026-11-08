"""Student-facing robot client. BLE runs in ble_worker.py on Windows or Linux."""
import atexit
import json
import math
from pathlib import Path
import subprocess
import sys
from threading import Condition, Lock, Thread
import time

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

class MecanumBLEClient:
    """Small synchronous interface to the separate BLE owner process."""
    def __init__(self, device_name="MP_BLE_Device", resolution=0.05):
        if not math.isfinite(resolution) or resolution <= 0:
            raise ValueError("resolution must be positive and finite")
        self.device_name = device_name
        self.resolution = resolution
        self.x = self.y = self.w = 0.0
        self.last_error = None
        self._process = None
        self._reader = None
        self._ready = False
        self._sequence = self._written = 0
        self._cached_velocity = None
        self._pipe_lock = Lock()
        self._status = Condition()
        atexit.register(self.disconnect)

    @property
    def is_connected(self):
        """True only after GATT discovery and the initial stop command finish."""
        return self._ready and self._process is not None and self._process.poll() is None

    def _read_status(self, process):
        try:
            for line in process.stdout:
                message = json.loads(line)
                with self._status:
                    if process is not self._process:
                        return
                    if 'ready' in message:
                        self._ready = message['ready']
                        self._cached_velocity = None
                    if 'written' in message:
                        self._written = max(self._written, message['written'])
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
        # ("Therian00", 0),
        ("Eq4", 0),
        # ("Eq2", 0),
        # ("Eq3", 1),
        # ("Eq1", 0),
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
        print("  Ctrl+C: Exit\n")

        while True:
            for car, gp_idx in cars:
                if car.is_connected:
                    car.set_velocity(get_user_cmd(gp_idx))

            time.sleep(0.02)  # update rate

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        for car, _ in cars:
            car.disconnect()  # The worker sends stop before disconnecting.
