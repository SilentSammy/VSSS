import asyncio
from bleak import BleakScanner, BleakClient as BleakClientLib
from threading import Thread, Lock
import time


# ============================================================================
# Utility Functions
# ============================================================================

def to_signed_byte(val):
    signed = max(-128, min(127, int(val * 128.0)))
    return signed & 0xFF


def pack_signed_bytes(values):
    return bytes([to_signed_byte(v) for v in values])


# ============================================================================
# Generic BLE Client
# ============================================================================

class BLEClient:

    def __init__(self, device_name):
        self.device_name = device_name
        self.device = None
        self.client = None

        # Connection management
        self._reconnect_enabled = False
        self._reconnect_thread = None

        # Thread safety
        self._write_lock = Lock()

        # ===== NEW: Control de envío =====
        self._last_data = None
        self._pending = None
        self._queued = None
        self._last_time = 0
        self._min_interval = 0.05  # 20 Hz

        # Background event loop
        self.loop = asyncio.new_event_loop()
        self.thread = Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    @property
    def is_connected(self):
        return self.client is not None and self.client.is_connected

    # ============================================================================
    # Public Methods
    # ============================================================================

    def connect(self):
        if self._reconnect_thread and self._reconnect_thread.is_alive():
            return
        self._reconnect_enabled = True
        self._reconnect_thread = Thread(target=self._reconnect_loop, daemon=True)
        self._reconnect_thread.start()

    def disconnect(self):
        self._reconnect_enabled = False
        if self.is_connected:
            future = asyncio.run_coroutine_threadsafe(self._async_disconnect(), self.loop)
            try:
                future.result(timeout=5.0)
            except Exception as e:
                print(f"Disconnect error: {e}")

    def write(self, uuid, data):
        """Write with cache + throttle + queue (latest wins)"""

        if not self.is_connected:
            return

        now = time.time()

        # Throttle
        if now - self._last_time < self._min_interval:
            self._queued = (uuid, data)
            return

        # Cache (avoid duplicates)
        if data == self._last_data:
            return

        # If not busy → send
        if self._pending is None or self._pending.done():
            self._pending = asyncio.run_coroutine_threadsafe(
                self._async_write(uuid, data),
                self.loop
            )
            self._pending.add_done_callback(lambda f: self._on_write_complete())

            self._last_data = data
            self._last_time = now

        else:
            # Busy → keep only latest
            self._queued = (uuid, data)

    # ============================================================================
    # Internal Logic
    # ============================================================================

    def _on_write_complete(self):
        if self._queued is not None:
            uuid, data = self._queued
            self._queued = None
            self.write(uuid, data)

    def _reconnect_loop(self):
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

    async def _async_find_device(self):
        print(f"Scanning for {self.device_name}...")
        devices = await BleakScanner.discover(timeout=10.0)
        device = next((d for d in devices if d.name == self.device_name), None)
        if not device:
            print("✗ Device not found")
            return None
        print(f"✓ Found at {device.address}")
        return device

    async def _async_connect(self):
        self.device = await self._async_find_device()
        if not self.device:
            raise Exception("Device not found")

        print("Connecting...")
        self.client = BleakClientLib(self.device.address)
        await self.client.connect()

        if not self.client.is_connected:
            raise Exception("Connection failed")

        print("✓ Connected\n")

    async def _async_disconnect(self):
        if self.client:
            await self.client.disconnect()
            print("Disconnected.")

    async def _async_write(self, uuid, data):
        if not self.is_connected:
            return
        await self.client.write_gatt_char(uuid, data)