import numpy as np
import asyncio
from bleak import BleakScanner, BleakClient
from threading import Thread
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
    import combined_input as inp

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
    def __init__(self, device_name="MP_BLE_Device", resolution=0.05):
        self.device_name = device_name
        self.device = None
        self.client = None
        
        # Velocity control fields
        self.x = 0.0
        self.y = 0.0
        self.w = 0.0
        
        # Control resolution - round commands to nearest multiple
        self.resolution = resolution
        
        # UUID for combined velocity characteristic
        self.uuid_velocity = "12345678-1234-5678-1234-56789abcdef1"
        
        # Cache last sent values to avoid redundant BLE writes
        self._cached_velocity = None
        
        # Track pending non-blocking writes and queued values
        self._pending = None
        self._queued = None

        # Connection management
        self._reconnect_enabled = False
        self._reconnect_thread = None

        # Start background event loop
        self.loop = asyncio.new_event_loop()
        self.thread = Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
    
    # Internal async methods
    async def _async_find_device(self):
        """Scan for and return the target BLE device"""
        print(f"Scanning for {self.device_name}...")
        devices = await BleakScanner.discover(timeout=10.0)
        device = next((d for d in devices if d.name == self.device_name), None)
        if not device:
            print(f"✗ Device not found")
            return None
        print(f"✓ Found at {device.address}")
        return device
    
    async def _async_connect(self):
        """Connect to the BLE device"""
        self.device = await self._async_find_device()
        if not self.device:
            raise Exception("Device not found")
        
        print(f"Connecting...")
        self.client = BleakClient(self.device.address)
        await self.client.connect()
        if not self.client.is_connected:
            raise Exception("Connection failed")
        print(f"✓ Connected\n")
    
    async def _async_disconnect(self):
        """Disconnect from the BLE device"""
        if self.client:
            await self.client.disconnect()
            print("Disconnected.")

    @property
    def is_connected(self) -> bool:
        """True if currently connected to the BLE device."""
        return self.client is not None and self.client.is_connected

    def _reconnect_loop(self):
        """Background thread: keeps attempting to connect while enabled."""
        while self._reconnect_enabled:
            if not self.is_connected:
                try:
                    future = asyncio.run_coroutine_threadsafe(self._async_connect(), self.loop)
                    future.result()
                except Exception as e:
                    print(f"Connection failed ({e}). Retrying in 3s...")
                    time.sleep(3)
            else:
                time.sleep(1)

    async def _async_send_velocity(self, x, y, w, force=False):
        """Send velocity command (3 bytes: x, y, w)"""
        if not self.is_connected:
            return
        velocity_bytes = (to_byte(x), to_byte(y), to_byte(w))
        
        # Check cache to avoid redundant writes (unless forced)
        if not force and self._cached_velocity == velocity_bytes:
            return  # No change, skip write
        
        await self.client.write_gatt_char(self.uuid_velocity, bytes(velocity_bytes))
        self._cached_velocity = velocity_bytes
    
    # Public synchronous methods
    def connect(self):
        """Start background connection and auto-reconnect loop (non-blocking)."""
        if self._reconnect_thread and self._reconnect_thread.is_alive():
            return
        self._reconnect_enabled = True
        self._reconnect_thread = Thread(target=self._reconnect_loop, daemon=True)
        self._reconnect_thread.start()
    
    def disconnect(self):
        """Stop reconnect loop and disconnect from the BLE device (blocking)."""
        self._reconnect_enabled = False
        if self.is_connected:
            future = asyncio.run_coroutine_threadsafe(self._async_disconnect(), self.loop)
            future.result()
    
    def send(self, force=False):
        """Send current velocity fields (x, y, w) to robot (non-blocking, queues if busy)
        
        Args:
            force: If True, bypasses cache and forces BLE write even if values unchanged
        """
        velocity_tuple = (self.x, self.y, self.w, force)
        
        if self._pending is None or self._pending.done():
            # Not busy - start new write
            self._pending = asyncio.run_coroutine_threadsafe(
                self._async_send_velocity(*velocity_tuple), 
                self.loop
            )
            # Add callback to process queued value when done
            self._pending.add_done_callback(lambda f: self._on_send_complete())
        else:
            # Busy - queue this value (replaces any previous queued value)
            self._queued = velocity_tuple
    
    def set_velocity(self, velocity, force=False):
        """Set velocity from dictionary and send command
        
        Args:
            velocity: Dictionary with keys 'x', 'y', 'w' (values -1.0 to 1.0)
            force: If True, bypasses cache and forces BLE write even if values unchanged
        """
        # Update internal state with rounding
        self.x = round(velocity.get('x', 0.0) / self.resolution) * self.resolution
        self.y = round(velocity.get('y', 0.0) / self.resolution) * self.resolution
        self.w = round(velocity.get('w', 0.0) / self.resolution) * self.resolution
        
        self.send(force=force)
    
    def _on_send_complete(self):
        """Callback when send completes - send queued values if exist"""
        if self._queued is not None:
            x, y, w, force = self._queued
            self._queued = None  # Clear queue
            # Update fields and send
            self.x = x
            self.y = y
            self.w = w
            self.send(force=force)
    
    def stop(self):
        """Stop all motors (blocking)"""
        future = asyncio.run_coroutine_threadsafe(
            self._async_send_velocity(0.0, 0.0, 0.0, force=True), 
            self.loop
        )
        future.result()
        # Update internal state
        self.x = 0.0
        self.y = 0.0
        self.w = 0.0

if __name__ == "__main__":
    import time

    print("Mecanum BLE Manual Control Demo")
    print("================================\n")

    # Each entry: (device_name, gamepad_index)
    # Add more tuples to control multiple cars simultaneously.
    CAR_CONFIG = [
        ("Eq2", 0),
        # ("Therian01", 1),
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
            car.stop()
            car.disconnect()
