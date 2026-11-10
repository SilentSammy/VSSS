import bluetooth
import time

# Global BLE setup
DEVICE_NAME = "MP_BLE_Device"
ADVERTISING_INTERVAL_US = 20000
HEARTBEAT_TIMEOUT_MS = 1500
CONNECT_TIMEOUT_MS = 10000  # Allow time for the laptop's initial GATT discovery.
_ble = bluetooth.BLE()
_ble.active(True)
_SERVICE_UUID_LE = bytes.fromhex('f0debc9a785634127856341278563412')

# Global state
control_callbacks = {}  # Dictionary mapping UUID strings to callback functions
initial_values = {}
_connections = set()    # Set of connected devices
_handles = {}           # Dictionary mapping UUID strings to characteristic handles
_deadlines = {}
_running = False
disconnect_callback = None  # main.py uses this to stop the motors.


def _release(conn_handle):
    if conn_handle in _connections:
        _connections.discard(conn_handle)
        _deadlines.pop(conn_handle, None)
        if disconnect_callback is not None:
            disconnect_callback()


def poll():
    """Call from the main loop: stop and disconnect if heartbeats disappear."""
    now = time.ticks_ms()
    for conn_handle in tuple(_connections):
        if time.ticks_diff(now, _deadlines[conn_handle]) >= 0:
            _release(conn_handle)  # Stop motors before releasing the BLE link.
            _ble.gap_disconnect(conn_handle)


def set_value(uuid, value, notify=False):
    """Update a readable characteristic, optionally notifying connected clients."""
    handle = _handles.get(uuid)
    if handle is None:
        raise ValueError("Unknown BLE characteristic")
    value = bytes(value)
    _ble.gatts_write(handle, value)
    if notify:
        for conn_handle in tuple(_connections):
            try:
                _ble.gatts_notify(conn_handle, handle, value)
            except OSError:
                pass

def _ble_irq(event, data):
    """Handle BLE events through interrupts"""
    if event == 1:  # _IRQ_CENTRAL_CONNECT
        conn_handle, _, _ = data
        if not _running or _connections:
            _ble.gap_disconnect(conn_handle)
            return
        _connections.add(conn_handle)
        _deadlines[conn_handle] = time.ticks_add(time.ticks_ms(), CONNECT_TIMEOUT_MS)
        print('Client connected')
        
    elif event == 2:  # _IRQ_CENTRAL_DISCONNECT
        conn_handle, _, _ = data
        _release(conn_handle)
        print('Client disconnected')
        if _running and not _connections:
            _advertise()
        
    elif event == 3:  # _IRQ_GATTS_WRITE
        conn_handle, value_handle = data
        if conn_handle in _connections:
            # Find which UUID this handle belongs to
            for uuid, handle in _handles.items():
                if handle == value_handle:
                    payload = _ble.gatts_read(value_handle)
                    if uuid in control_callbacks:
                        # Call the corresponding callback with the full data array
                        result = control_callbacks[uuid](payload)
                        if result is not None:
                            _ble.gatts_write(value_handle, result)
                        _deadlines[conn_handle] = time.ticks_add(
                            time.ticks_ms(), HEARTBEAT_TIMEOUT_MS)
                    break

def _advertise(interval_us=None):
    """Start advertising the BLE service"""
    if interval_us is None:
        interval_us = ADVERTISING_INTERVAL_US
    name = DEVICE_NAME.encode()
    # Put the service identity in the primary advertisement so a computer can
    # recognize a never-before-seen robot without relying on its GATT cache.
    adv_data = bytearray([0x02, 0x01, 0x06, 0x11, 0x07]) + _SERVICE_UUID_LE
    response_data = bytearray([len(name) + 1, 0x09]) + name
    try:
        _ble.gap_advertise(interval_us, adv_data, resp_data=response_data)
    except TypeError:
        # Compatibility with older MicroPython builds.
        fallback = bytearray([0x02, 0x01, 0x06, len(name) + 1, 0x09]) + name
        _ble.gap_advertise(interval_us, fallback)


def stop():
    """Stop the BLE server and clean up"""
    global _running
    _running = False
    # Stop advertising
    _ble.gap_advertise(None)
    
    # Remove IRQ handler
    _ble.irq(None)
    
    # Disconnect any active connections
    for conn_handle in _connections.copy():  # Use copy since we're modifying the set
        _release(conn_handle)
        _ble.gap_disconnect(conn_handle)
    
    print("BLE Server stopped")

def start():
    """Initialize and start the BLE server"""
    global _handles, _running
    
    # Ensure clean state
    stop()
    # Also clear controller state left behind by serial/IDE soft resets.
    _ble.active(False)
    _ble.active(True)
    
    if not control_callbacks:
        raise ValueError("No callbacks defined!")
        
    _ble.config(gap_name=DEVICE_NAME)
    
    # Create characteristics for each callback
    characteristics = []
    uuids = sorted(control_callbacks.keys())  # Sort for consistent ordering
    
    for uuid_str in uuids:
        uuid = bluetooth.UUID(uuid_str)
        characteristics.append(
            (uuid, bluetooth.FLAG_READ | bluetooth.FLAG_WRITE | bluetooth.FLAG_NOTIFY)
        )
    
    # Register all characteristics under one service
    service_handles = _ble.gatts_register_services(((
        bluetooth.UUID('12345678-1234-5678-1234-56789abcdef0'),  # Service UUID
        tuple(characteristics),
    ),))
    
    # Map UUIDs to their handles and initialize to 0
    _handles.clear()
    for i, uuid_str in enumerate(uuids):
        handle = service_handles[0][i]
        _handles[uuid_str] = handle
        value = initial_values.get(uuid_str, bytes([0]))
        _ble.gatts_set_buffer(handle, max(20, len(value)))
        _ble.gatts_write(handle, value)
        
    # Set up event handler and start advertising
    time.sleep_ms(1000)  # Let the ESP32 controller settle after rebuilding GATT (boot only).
    _running = True
    _ble.irq(_ble_irq)
    _advertise()
    
    print(f"BLE Server started with {len(control_callbacks)} controls")
    print(f"Advertising as {DEVICE_NAME}")

# Conversion helpers
def to_norm(value):
    """Convert 0-255 to 0.0-1.0 float"""
    return value / 256.0

def to_bipolar(value):
    """Convert signed int8 (0-255) to -1.0-1.0 float using two's complement"""
    # Convert unsigned byte to signed int8
    signed = value if value < 128 else value - 256
    # Normalize to -1.0 to 1.0
    return signed / 128.0
