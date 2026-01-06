import asyncio
from bleak import BleakScanner, BleakClient

def to_byte(val):
    """Convert -1.0 to +1.0 range to 0-255 byte using two's complement"""
    signed = max(-128, min(127, int(val * 128.0)))
    return signed & 0xFF

class MecanumBLEClient:
    def __init__(self, device_name="MP_BLE_Device"):
        self.device_name = device_name
        self.device = None
        self.client = None
        
        # UUIDs matching the MecanumCar server
        self.uuid_x = "12345678-1234-5678-1234-56789abcdef1"
        self.uuid_y = "12345678-1234-5678-1234-56789abcdef2"
        self.uuid_w = "12345678-1234-5678-1234-56789abcdef3"
        
        # Cache last sent values to avoid redundant BLE writes
        self._cached_char_values = {}
        
        # Track pending non-blocking writes
        self._pending_x = None
        self._pending_y = None
        self._pending_w = None
    
    async def find_device(self):
        """Scan for and return the target BLE device"""
        print(f"Scanning for {self.device_name}...")
        devices = await BleakScanner.discover(timeout=10.0)
        device = next((d for d in devices if d.name == self.device_name), None)
        if not device:
            print(f"✗ Device not found")
            return None
        print(f"✓ Found at {device.address}")
        return device
    
    async def connect(self):
        """Connect to the BLE device"""
        self.device = await self.find_device()
        if not self.device:
            raise Exception("Device not found")
        
        print(f"Connecting...")
        self.client = BleakClient(self.device.address)
        await self.client.connect()
        if not self.client.is_connected:
            raise Exception("Connection failed")
        print(f"✓ Connected\n")
    
    async def disconnect(self):
        """Disconnect from the BLE device"""
        if self.client:
            await self.client.disconnect()
            print("Disconnected.")
    
    async def set_char(self, char_uuid, value, force=False):
        """Write to characteristic with caching to avoid redundant writes"""
        if not force and self._cached_char_values.get(char_uuid) == value:
            return  # No change, skip write
        await self.client.write_gatt_char(char_uuid, bytes([value]))
        self._cached_char_values[char_uuid] = value
    
    async def set_x(self, value):
        """Set X-axis (forward/backward) value (-1.0 to 1.0)"""
        await self.set_char(self.uuid_x, to_byte(value))
    
    async def set_y(self, value):
        """Set Y-axis (strafe) value (-1.0 to 1.0)"""
        await self.set_char(self.uuid_y, to_byte(value))
    
    async def set_w(self, value):
        """Set W-axis (rotation) value (-1.0 to 1.0)"""
        await self.set_char(self.uuid_w, to_byte(value))
    
    def send_x(self, value):
        """Send X-axis command (non-blocking, skips if previous write pending)"""
        if self._pending_x is None or self._pending_x.done():
            self._pending_x = asyncio.create_task(self.set_x(value))
    
    def send_y(self, value):
        """Send Y-axis command (non-blocking, skips if previous write pending)"""
        if self._pending_y is None or self._pending_y.done():
            self._pending_y = asyncio.create_task(self.set_y(value))
    
    def send_w(self, value):
        """Send W-axis command (non-blocking, skips if previous write pending)"""
        if self._pending_w is None or self._pending_w.done():
            self._pending_w = asyncio.create_task(self.set_w(value))
    
    async def stop(self):
        """Stop all motors (blocking)"""
        await self.set_x(0)
        await self.set_y(0)
        await self.set_w(0)

async def control_loop():
    """Run control loop with keyboard/gamepad input"""
    import combined_input as inp
    car = MecanumBLEClient()
    await car.connect()
    
    try:
        print("Control active! Use WASD/gamepad to control.")
        print("  W/S or Left Stick Y: Forward/Backward (X-axis)")
        print("  A/D or Left Stick X: Strafe Left/Right (Y-axis)")
        print("  Q/E or Right Stick X: Rotate (W-axis)")
        print("  ESC: Exit\n")
        

        while True:
            scale = 1.0 if inp.is_pressed('c') else 0.5  # 'C' key for full speed
            
            # X-axis: forward/backward (W/S or left stick Y)
            x = inp.get_bipolar_ctrl('w', 's', 'LY') * scale
            
            # Y-axis: strafe (A/D or left stick X)
            y = inp.get_bipolar_ctrl('d', 'a', 'LX') * scale
            
            # W-axis: rotation (Q/E or right stick X)
            w = inp.get_bipolar_ctrl('e', 'q', 'RX') * scale
            
            await car.set_x(x)
            await car.set_y(y)
            await car.set_w(w)
            
            if inp.is_pressed('Key.esc'):
                break
            
            await asyncio.sleep(0.02)
            
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        await car.disconnect()

if __name__ == "__main__":
    try:
        asyncio.run(control_loop())
        # asyncio.run(test_seq())
    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as e:
        print(f"Error: {e}")
