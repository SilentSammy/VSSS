"""Generic BLE Client with utility functions for BLE communication

This module provides:
- BLEClient: Generic class for connecting and writing to any BLE device
- Packing utilities: Helper functions to convert Python values to bytes
"""

import asyncio
from bleak import BleakScanner, BleakClient as BleakClientLib
from threading import Thread, Lock
import time


# ============================================================================
# Utility Functions for Byte Packing
# ============================================================================

def to_signed_byte(val):
    """Convert single float [-1.0, 1.0] to byte [0-255] using two's complement
    
    Args:
        val: Float value in range [-1.0, 1.0]
        
    Returns:
        int: Byte value [0-255] where 128 is zero, 0 is -1.0, 255 is ~1.0
        
    Example:
        >>> to_signed_byte(0.0)
        128
        >>> to_signed_byte(1.0)
        255
        >>> to_signed_byte(-1.0)
        0
    """
    signed = max(-128, min(127, int(val * 128.0)))
    return signed & 0xFF


def pack_signed_bytes(values):
    """Pack list of floats [-1.0, 1.0] into signed bytes (two's complement)
    
    Args:
        values: List/tuple of floats in range [-1.0, 1.0]
        
    Returns:
        bytes: Packed bytes ready for BLE transmission
        
    Example:
        >>> pack_signed_bytes([0.5, -0.3, 0.0])
        b'\\xc0\\xd9\\x80'
    """
    return bytes([to_signed_byte(v) for v in values])


def to_unsigned_byte(val, min_val=0.0, max_val=1.0):
    """Convert float to unsigned byte [0-255]
    
    Args:
        val: Float value to convert
        min_val: Minimum value (maps to 0)
        max_val: Maximum value (maps to 255)
        
    Returns:
        int: Byte value [0-255]
    """
    normalized = (val - min_val) / (max_val - min_val)
    return max(0, min(255, int(normalized * 255)))


def pack_unsigned_bytes(values, min_val=0.0, max_val=1.0):
    """Pack list of floats into unsigned bytes [0-255]
    
    Args:
        values: List/tuple of floats
        min_val: Minimum value (maps to 0)
        max_val: Maximum value (maps to 255)
        
    Returns:
        bytes: Packed bytes ready for BLE transmission
    """
    return bytes([to_unsigned_byte(v, min_val, max_val) for v in values])


# ============================================================================
# Generic BLE Client
# ============================================================================

class BLEClient:
    """Generic BLE client for connecting and writing to BLE devices
    
    Handles connection management, auto-reconnect, and async event loop.
    Students can use this to communicate with any BLE device by specifying
    device name and characteristic UUIDs.
    
    Example:
        >>> client = BLEClient("Therian00")
        >>> client.connect()
        >>> uuid = "12345678-1234-5678-1234-56789abcdef1"
        >>> client.write(uuid, pack_signed_bytes([0.5, -0.3, 0.0]))
        >>> client.disconnect()
    """
    
    def __init__(self, device_name):
        """Initialize BLE client
        
        Args:
            device_name: Name of the BLE device to connect to
        """
        self.device_name = device_name
        self.device = None
        self.client = None
        
        # Connection management
        self._reconnect_enabled = False
        self._reconnect_thread = None
        
        # Thread safety for write operations
        self._write_lock = Lock()
        
        # Start background event loop
        self.loop = asyncio.new_event_loop()
        self.thread = Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
    
    @property
    def is_connected(self) -> bool:
        """True if currently connected to the BLE device"""
        return self.client is not None and self.client.is_connected
    
    def connect(self):
        """Start background connection and auto-reconnect loop (non-blocking)
        
        This method returns immediately. Connection happens in the background.
        Check is_connected property to verify connection status.
        """
        if self._reconnect_thread and self._reconnect_thread.is_alive():
            return
        self._reconnect_enabled = True
        self._reconnect_thread = Thread(target=self._reconnect_loop, daemon=True)
        self._reconnect_thread.start()
    
    def disconnect(self):
        """Stop reconnect loop and disconnect from BLE device (blocking)"""
        self._reconnect_enabled = False
        if self.is_connected:
            future = asyncio.run_coroutine_threadsafe(self._async_disconnect(), self.loop)
            try:
                future.result(timeout=5.0)
            except Exception as e:
                print(f"Disconnect error: {e}")
    
    def write(self, uuid, data):
        """Write data to a BLE characteristic (non-blocking)
        
        Args:
            uuid: UUID string of the characteristic to write to
            data: bytes object to send
        """
        if not self.is_connected:
            return
        
        with self._write_lock:
            future = asyncio.run_coroutine_threadsafe(
                self._async_write(uuid, data), 
                self.loop
            )
            # Add error callback to catch write failures
            future.add_done_callback(self._on_write_complete)
    
    # Internal methods
    def _reconnect_loop(self):
        """Background thread: keeps attempting to connect while enabled"""
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
        self.client = BleakClientLib(self.device.address)
        await self.client.connect()
        if not self.client.is_connected:
            raise Exception("Connection failed")
        print(f"✓ Connected\n")
    
    async def _async_disconnect(self):
        """Disconnect from the BLE device"""
        if self.client:
            await self.client.disconnect()
            print("Disconnected.")
    
    async def _async_write(self, uuid, data):
        """Write data to BLE characteristic"""
        if not self.is_connected:
            return
        await self.client.write_gatt_char(uuid, data)
    
    def _on_write_complete(self, future):
        """Callback to handle write completion and catch errors"""
        try:
            future.result()
        except Exception as e:
            print(f"Write error: {e}")


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    print("BLE Client Demo")
    print("===============\n")
    
    # Connect to device
    client = BLEClient("Therian00")
    client.connect()
    
    # Wait for connection
    print("Waiting for connection...")
    while not client.is_connected:
        time.sleep(0.5)
    
    # Send some test commands
    uuid_velocity = "12345678-1234-5678-1234-56789abcdef1"
    
    print("Sending velocity commands...")
    for i in range(5):
        x = 0.5 if i % 2 == 0 else -0.5
        client.write(uuid_velocity, pack_signed_bytes([x, 0.0, 0.0]))
        time.sleep(1)
    
    # Stop and disconnect
    client.write(uuid_velocity, pack_signed_bytes([0.0, 0.0, 0.0]))
    time.sleep(0.5)
    client.disconnect()
    print("Done.")
