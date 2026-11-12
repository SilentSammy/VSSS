# VSSS student blocks - live robot control

Students build programs with Blockly, view the generated Python, and run them
against the real VSSS `Car` API. The track view and teaching simulator have been
removed. There is no simulation mode in the application.

## Launch

From the VSSS repository root, use the existing VSSS Python environment:

```powershell
.\.venv\Scripts\python.exe blockly_python_demo\server.py
```

Open **http://127.0.0.1:8765**. Join the `embedded` lab Wi-Fi and enable Bluetooth.
Make sure the camera service is publishing pose and the selected marker is visible.
Check the car number in the `with car` block, then click **Run live**.
The starter car program contains motion commands. Nothing connects or moves merely
from opening the editor; execution begins only when Run live is clicked.

If needed, install the parent project's Python requirements in that environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For a fresh checkout, install the pinned Blockly/browser-test dependencies once:

```powershell
cd blockly_python_demo
npm.cmd ci
```

No account, sign-in, API key, or cloud service is required. The editor's assets
are local and work offline. Lab Wi-Fi is still required for camera feedback.
Use `--port 8767` to change the editor port. Ctrl+C stops the server and cleans up
the active run. Run the server with the Python interpreter that has `bleak` installed.

## Controls

- **Run live** starts the current program. There is one active run per server.
- **Stop** interrupts the program, sends a stop command, and disconnects the car.
  It remains usable during long waits, camera waits, and endless Python loops.
- **Car example / Python example** restore the robot or original print/loop demo.
- **Save blocks / Open blocks** save and restore local JSON workspaces.
- **Export Python** downloads ordinary `Car` Python code for use outside the editor.

Output streams into the page while a program runs. Run is disabled until the
previous program finishes cleanup. Blocks can be edited during a run; edits apply
to the next run. Nothing is autosaved.

## Student block mapping

| Block | Generated Python | Meaning |
| --- | --- | --- |
| with car N / do | `with Car(N) as car:` | Connect to carN and clean up on exit |
| velocity forward / left / turn CCW | `car.set_velocity(x=..., y=..., w=...)` | Normalized robot-frame commands, each in [-1, 1] |
| go to field X / Y / heading | `car.go_to(x, y, angle)` | Field metres and heading degrees; onboard waypoint control |
| wait N seconds | `sleep(N)` | Wait actual seconds while an active command continues |
| stop car | `car.stop()` | Stop motion and continue the rest of the program |
| wait for camera position | `car.wait_for_position(timeout=...)` | Wait for fresh feedback or fail on timeout |
| camera pose | `car.pose` | Pose(x, y, angle), or None if missing/stale |
| camera position | `car.position` | (x, y), or None if missing/stale |
| camera position available? | `car.pose is not None` | Check pose freshness and marker visibility |
| robot status | `car.read_status()` | Read actual onboard mode, flags, fault and pose |

Use one `with car` block per program. Place car and sensor blocks inside it.
Loops, variables, print, arithmetic and comparisons can be used inside the session.
A waypoint returns when acknowledged, **not when the car arrives**. A wait is a
duration, not an arrival check. Exiting the session stops/disconnects the car.
Normalized velocity is not a physical speed in metres per second.

## Live execution and stopping

The browser sends generated Python to a local threaded HTTP server. A disposable
Python subprocess executes the student logic. Its Car proxy sends requests over
private parent/child pipes to the supervisor, which owns the **real Car instance**
and its BLE connection. The supervisor calls the existing public Car methods;
real GPS/UDP supplies feedback. No hardware connection is held by the disposable
student interpreter, so killing an endless loop does not orphan a moving BLE client.

The supervisor stops/disconnects on normal completion, runtime/API error, Stop,
server shutdown, and these conditions:

- The owning browser closes or stops sending heartbeats for 5 seconds.
- The run exceeds 120 seconds, including connection setup.
- The robot disconnects or its connection generation changes; motion never resumes
  automatically after reconnection.
- Camera pose becomes missing/stale during motion. Fresh pose is also required
  before any nonzero velocity or waypoint command is accepted.

Stop first revokes further commands and ends the student process. The supervisor
then sends stop and closes BLE and the camera receiver. The UI shows **Stopping**
until cleanup completes. An already-running BLE operation can take its existing
API timeout to return before cleanup; stopping during initial discovery may wait
for the connection attempt to finish (normally bounded by Car's 15-second timeout).
Stop/cleanup errors appear in the output rather than reporting successful completion.

This is a software stop through BLE, not a physical emergency-stop circuit. If the
link or server process is lost, stopping also relies on the existing firmware
lease/timeout behavior. This editor does not add field-boundary protection or a
new onboard controller. The track limits and firmware behavior remain unchanged.

The local runner accepts a restricted Python subset, limits output to 16,000
characters, and accepts only the Car/sleep imports and initial student API methods.
It is for a trusted local classroom machine, not an untrusted-code hosting service.
Do not expose the localhost server on a network.

## Exported Python

The export remains ordinary Python:

```python
from car import Car
from time import sleep

with Car(1) as car:
    car.wait_for_position(timeout=10)
    car.set_velocity(x=0.2, y=0, w=0)
    sleep(1)
    car.stop()
```

To run an export separately, place it in the repository root or install VSSS with
`pip install -e .`, and run it with the configured environment. Exported scripts
can move hardware and do not include the editor's supervisor, heartbeat, runtime
limit or additional pose guard. The existing Car/transport/firmware rules apply.

## Verification

```powershell
cd blockly_python_demo
npm.cmd test
```

Thirteen Python tests cover real-duration command execution, stop during waits or
CPU loops, runtime errors, heartbeat expiry, run limits, missing/stale pose,
disconnect/reconnection, overlapping runs, stale Stop requests, cancellation during
connection and camera waits, output limits, and failed stop acknowledgement.

One integration test uses the actual `Car`, actual BLE worker subprocess and actual
UDP receiver, with the repository's fake radio and synthetic localhost camera
packets. It verifies an encoded velocity command, a subsequent zero command, and
BLE disconnection. No radio adapter or physical robot is used by these tests.

Browser checks use `test_http_server.py`, an explicit test-only fake-I/O fixture.
They cover the live controls, streamed output, real-time waits, Stop, page closure,
Python export, orphan-block rejection, edits, save/load, runtime errors and request
guards, and assert that the simulator view is absent. Export is also checked against
the real Car class with mocked I/O. `server.py` always selects the real Car path.
Physical motion and live radio delivery still require a supervised lab check.

Key files: `vsss_blocks.js` (blocks/generators), `app.js` (editor/live controls),
`server.py` (HTTP), `live_runtime.py` (hardware owner/watchdog), `worker.py` (student
interpreter). Existing parent Car, GPS, transport and firmware files are unchanged.
