# BLE robot control

The client works the same way on Windows and Linux. The robot advertises as
`Eq4` by default; change the name in `mp/main.py` and the client's `CAR_CONFIG`
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

with MecanumBLEClient("Eq4") as robot:
    robot.set_velocity({"x": 0.2, "y": 0, "w": 0})
    time.sleep(1)
    robot.stop()
```

The context manager waits for readiness and always disconnects on exit.
The original nonblocking API also works:

```python
robot = MecanumBLEClient("Eq4")
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
BLE belongs to the worker. Use the public methods above. Earlier experimental
scripts that accessed these internals are historical measurements, not the
current client API.

## Tests

```shell
python -m unittest discover -s tests -v
```

Tests exercise real subprocesses and abrupt parent termination with a fake BLE
radio, plus a simulated MicroPython clock/radio for the firmware lease. They
require no adapter and never move a motor. All 12 tests pass on Windows/Python
3.11 and Ubuntu/Python 3.12. Actual BLE checks were also run against the flashed
board; Linux radio testing still requires a Bluetooth adapter on that computer.

Live testing also found that changing PWM frequency could exhaust the C3's
timer/channel pool. Commands now update all three axes together, and motors
release their old PWM allocation before changing frequency. Existing motor
pin assignments, frequency range, and direction conventions are preserved.
The board completed 100 rapidly changing mixed-axis commands after this fix.

`experiments/ble_upgrade/hardware_check.py` repeats live heartbeat/crash tests.
It deliberately sends nonzero velocities and is intended for an unmounted
board. Original board files were backed up in that directory's `device_backup`.
