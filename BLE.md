# BLE robot control

The client works the same way on Windows and Linux. The robot advertises as
`car1` by default; change the name in `mp/main.py` and the client's `CAR_CONFIG`
if needed. Linux needs a working BlueZ service and permission to use Bluetooth.

## Flash and run

Copy these three files to the ESP32's root directory, preserving the names:

- `mp/main.py` -> `main.py`
- `mp/ble_server.py` -> `ble_server.py`
- `mp/mecanum.py` -> `mecanum.py`

Reset the ESP32 after copying all three. Run the existing manual-control demo:

```shell
python mecanum_client.py
```

It retains the existing WASD/gamepad controls. `Ctrl+C` stops and disconnects.
The two host files `mecanum_client.py` and `ble_worker.py` must stay together.
There are no new dependencies beyond Bleak (already in `requirements.txt`).
The board allows one second after GATT registration before its first
advertisement, required for reliable startup on the tested ESP32-C3 build.
This happens at firmware startup, not each time a student connects.

For a student script:

```python
from mecanum_client import MecanumBLEClient
import time

with MecanumBLEClient("car1") as robot:
    robot.set_velocity({"x": 0.2, "y": 0, "w": 0})
    time.sleep(1)
    robot.stop()
```

The context manager waits for readiness and always disconnects on exit.
The original nonblocking API also works:

```python
robot = MecanumBLEClient("car1")
robot.connect()
try:
    robot.wait_until_connected(timeout=15)
    # Send commands here, or check robot.is_connected in a control loop.
finally:
    robot.disconnect()
```

Commands submitted before readiness are ignored. After a dropped connection,
the worker reconnects at zero velocity; the script must send a new movement.

## Three small pieces

1. `mecanum_client.py` is the student-facing interface. It converts velocity to
   bytes and writes a line of JSON to a private pipe. A reader thread receives
   readiness and write acknowledgements. It contains no Bluetooth event loop.
2. `ble_worker.py` is a separate process owning the actual Bleak connection.
   It discovers the first matching name, passes that device directly to Bleak,
   and sends only the newest velocity. It starts only when `connect()` is called;
   importing the module starts no process and needs no multiprocessing setup.
3. `mp/ble_server.py` receives commands. `mp/main.py` applies them to the motors
   and calls `ble_server.poll()` every 20 ms to enforce the heartbeat timeout.

The BLE payload is unchanged: three signed bytes for x, y, and rotation, using
characteristic `12345678-1234-5678-1234-56789abcdef1`. No extra service, pairing,
Windows-only service cache, or network connection is needed.

## Trim

The single trim characteristic is `12345678-1234-5678-1234-56789abcdef3`, with
READ/WRITE properties. Its payload is **36 bytes**: nine little-endian float32
values in row-major order. Rows are corrected outputs, columns are requested
inputs, ordered x/y/w. Diagonal gains are in [0, 2]; off-diagonal correction
terms in [-2, 2]; all entries must be finite. Defaults are the identity matrix.
The former three-gain characteristic ending in `def2` is removed.

Firmware computes `corrected = matrix * requested`, clips each result to [-1, 1],
then mixes wheel powers. No constant offset is added: zero command stays zero.
The matrix is kept in RAM until reboot. Invalid writes restore the previous
canonical value. The client verifies every write with a readback.

Use `Car.set_trim(matrix)` or `MecanumBLEClient.set_trim(matrix)` to replace the
matrix, or omit the argument to restore identity. The return value and
`trim_matrix` property expose the complete tuple of three rows. There is no
separate gain tuple, synchronization step, or legacy gain-write path. Firmware
without matrix support remains driveable but configuration reports an update
requirement. Deploy `mp/axis_trim.py` alongside the other firmware files.

The server reserves a 36-byte characteristic buffer with
[`gatts_set_buffer`](https://docs.micropython.org/en/latest/library/bluetooth.html#bluetooth.BLE.gatts_set_buffer)
because MicroPython's default incoming value limit is 20 bytes. Velocity commands
still use the original three-byte characteristic ending in `def1`.

## Cleanup and heartbeat

- The worker resends the current velocity every **200 ms**, even if unchanged.
  A velocity stays active until changed or stopped while the script is alive.
- A robot stops its motors and disconnects after **1500 ms** without a command.
  Initial connection setup gets **10000 ms** to allow GATT discovery.
- Normal `disconnect()` closes the pipe. The worker sends zero, disconnects
  BLE, and exits. There is no connection-retention period between scripts.
- If the script crashes or is forcefully terminated, closing its pipe has the
  same effect even though its Python cleanup code did not run.
- If the worker or laptop also disappears, the robot timeout releases control.
- `stop()` waits up to two seconds for acknowledgement of its forced zero
  command. BLE operations and worker shutdown have bounded timeouts.

Discovery still has a ten-second *maximum* for an absent robot, but returns
immediately when its advertisement matches. `is_connected` becomes true only
after GATT setup and an initial zero command complete. Parent scripts share no
global daemon, and one student's worker cannot keep running between scripts.

The old internal `client`, `loop`, and `thread` objects are no longer exposed;
BLE belongs to the worker. Use the public methods above.

## Tests

```shell
python -m unittest discover -s tests -v
```

Tests exercise real subprocesses and abrupt parent termination with a fake BLE
radio, plus a simulated MicroPython clock/radio for the firmware lease. They
require no adapter and never move a motor. Actual BLE checks were also run against the flashed
board; Linux radio testing still requires a Bluetooth adapter on that computer.

Live testing also found that changing PWM frequency could exhaust the C3's
timer/channel pool. Commands now update all three axes together, and motors
release their old PWM allocation before changing frequency. Existing motor
pin assignments, frequency range, and direction conventions are preserved.
The board completed 100 rapidly changing mixed-axis commands after this fix.
