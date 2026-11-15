# VSSS student blocks

A student toolbox for programming real VSSS cars. The default example is plain
Python printing and addition, so opening the editor and running that example does
not connect to hardware. There is no track simulator.

## Start

From the repository root:

```powershell
.\.venv\Scripts\python.exe blockly_python_demo\server.py
```

Open **http://127.0.0.1:8765**. Restart an older server and refresh the browser after
updating. The server uses the existing VSSS environment. A fresh copy needs the
parent Python requirements and `npm.cmd ci` in this folder. No sign-in or API key
is required. Use `--port 8767` if needed.

## Examples

Every requested example is available in the example selector and as a Python file:

| Example | Python file | Behavior |
| --- | --- | --- |
| Print and add | [01_print_and_add.py](examples/01_print_and_add.py) | Print a greeting and 7 + 5. No robot. |
| Variables and loops | [02_variables_and_loops.py](examples/02_variables_and_loops.py) | Add 2 to a running total five times and print 2, 4, 6, 8, 10. No robot. |
| Rotate back and forth | [03_rotate_back_and_forth.py](examples/03_rotate_back_and_forth.py) | Car 1: w=0.2 for one second, w=-0.2 for one second, repeat forever. No translation or GPS requirement. |
| Read car position | [04_read_car_position.py](examples/04_read_car_position.py) | Print marker 1's Pose(x, y, angle) every .25 seconds, or None if missing/stale. GPS only: no BLE connection or motion commands. |
| Alternate waypoints | [05_alternate_waypoints.py](examples/05_alternate_waypoints.py) | Car 1: alternate (0, .5, 90) and (0, -.5, 90) forever, waiting for arrival at each. |
| Partial teleop | [06_partial_teleop.py](examples/06_partial_teleop.py) | W/S drive at +.5/-.5 in X; gamepad 0 RX drives yaw at half scale. Y stays zero for students to complete. |
| Read gamepad | [07_read_gamepad.py](examples/07_read_gamepad.py) | Print A/B/X/Y buttons, LX/LY/RX/RY joysticks, and LT/RT triggers on gamepad 0 every .25 seconds. No car or GPS. |

In partial teleop, releasing W/S (or holding both) gives X=0, and centering RX
gives yaw=0. The loop runs every .05 seconds until Stop or Ctrl+C. Input is read
on the server computer. No GPS is needed.

The rotation and waypoint examples move a physical car when Run live is clicked.
Check the car number and available space first. Waypoints use field X/Y metres
and heading degrees; yaw velocity is a normalized CCW command, not degrees/s.
The waypoint example checks onboard feedback/fault state and fails after ten
seconds without arrival rather than repeatedly overwriting an unfinished target.

Use **Stop** in the editor, or Ctrl+C for a separately executed Python example.
There is no default two-minute run cutoff. Infinite examples continue until Stop,
page closure, loss of the browser heartbeat, robot disconnection, or an error.
The output pane keeps the latest 16,000 characters instead of ending a long
position-reading program when the output fills up.

For example, run a standalone script from the repository root with:

```powershell
.\.venv\Scripts\python.exe blockly_python_demo\examples\03_rotate_back_and_forth.py
```

For standalone robot scripts, VSSS must be installed in that environment
(`pip install -e .`). The two basic Python scripts need no VSSS dependency.
Other examples retained under More examples cover two cars, two gamepads,
keyboard control and field observation.

## Student toolbox

Core blocks cover selecting/naming cars, velocity, waypoints, stop, position and
heading, other cars and balls, held keyboard/gamepad buttons, normalized axes,
print, variables, arithmetic, conditions, loops, waits and lists.

Multiple named cars can be commanded in one loop or placed in a list for a
for-each loop. Commands are sequential, while each car independently continues
its most recent command. Velocity uses BLE only. Field reads start GPS lazily;
waypoint control needs the robot's separate onboard position reception.

**Advanced lessons** groups input edges/toggles, combined keyboard/gamepad
controls, clock/timeouts, waiting for camera position and readable robot checks:

- target reached?
- onboard position available?
- robot fault?
- wait until the target is reached (with a timeout)

Students select X, Y or angle from pose values. They do not need raw firmware
flags, packet counters or transport state. The wait-for-arrival block generates
ordinary Python polling code with fault/freshness/timeout checks.

Removed from both the student toolbox and approved live execution surface:

- Trim matrices and controller configuration/readbacks.
- Reboot, telemetry recovery, connection-generation and individual-disconnect APIs.
- GPS ports, subscriptions, stale-time configuration, shutdown and packet metadata.
- Raw axes, global inversion settings and keyboard-availability plumbing.
- Generic JSON configuration blocks.

These operations remain available to the instructor through the normal VSSS
Python API and command-line tools. Parent VSSS modules are unchanged. Cleanup
remains an internal supervisor responsibility, not a student configuration block.

## Input and cleanup

Input reads use `vsss.input` on the server computer. Keyboard names include `w`
and `Key.space`. Gamepads are numbered from 0; helper blocks generate names such
as `LX0`, `LY1`, `A1` and `L31`. The input parser currently supports device suffixes
0-9. The module's native watchers are loaded only on the first input read and are
reused for the server lifetime. No browser input emulation is added.

Robot/input/field programs export with a synchronous `@run` entry point. Basic
print/arithmetic/variable examples export as plain Python. In live mode a disposable
interpreter sends approved requests to the supervisor, which owns the real cars.
Stop kills the student program and stops/disconnects all cars independently.
A blocked call on one car does not delay sending Stop to the others; the blocked
car must finish its existing API timeout before its own cleanup completes.

Closing the owning page also stops the program, with a five-second heartbeat
expiry as fallback. Losing/replacing any car's connection ends the entire run.
GPS is released after run cleanup. Stop/cleanup errors are displayed while cleanup
of other cars is still attempted. This is a software BLE stop, not a physical
emergency-stop circuit; existing provider/firmware stopping behavior still applies.

Exports and separately executed example files use native `vsss.run` cleanup but
do not include the editor's independent Stop button or browser heartbeat.

## Maintenance and checks

`api_registry.py` is the explicit student approval list. It reads current parameter
names/defaults/docstrings from the approved source files without importing device
modules. New APIs still need approval, and semantic changes can need an adapter.
Restart the server and refresh after Python API changes. Old motion-only single-car
block JSON is supported; saved instructor/configuration blocks are no longer
approved and need to be removed from student workspaces.

From this folder:

```powershell
npm.cmd test
```

Tests use fake devices and actual input API functions with seeded state. They
check multi-car routing/cleanup, instructor-operation rejection, source-driven
schema, each Python example, opposite rotation commands, repeated waypoint targets,
position-only observation, generated blocks, export, save/load, Stop and page
closure. Physical Bluetooth, gamepads and motion still need a supervised lab check.
Save blocks explicitly before closing; there is no autosave. This localhost runner
is intended for a trusted classroom computer, not untrusted-code hosting.
