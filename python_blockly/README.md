# VSSS student blocks

A student toolbox for programming VSSS cars in simulation or live. The default
environment is `car1_ball1` (1 car 1 ball); opening the page does not start hardware. Choose
an environment or press Run to start. Generated Python uses the same Car/GPS API.

## Start

From the repository root:

```powershell
.\.venv\Scripts\python.exe python_blockly\server.py
```

Open **http://127.0.0.1:8765**. Restart an older server and refresh the browser after
updating. The server uses the existing VSSS environment. A fresh copy needs the
parent Python requirements and `npm.cmd ci` in this folder. No sign-in or API key
is required. Use `--port 8767` if needed.
`--endpoint PATH --provider-port 0` gives the editor an isolated provider endpoint
and a free RPC port; student code and the embedded plot use that endpoint too.

The Environment dropdown offers exactly Live, 1 car, 1 car 1 ball, 2 cars 1 ball,
4 cars 1 ball, goalie easy and goalie hard. Simulation uses localhost RPC and
does not need lab Wi-Fi or the live UDP feed. Failed editor requests time out,
show an error and allow retry; the scenario controls recover after a lost run's
five-second heartbeat lease expires. Initial environment loading retries too.
Selecting a scenario starts the normal VSSSSim provider, or resets the running
simulator over RPC. Start/reset reloads the current scenario. Live starts the
normal GPSLive/CarCmdLive provider and may control physical cars. These are the
same implementations as `vsss-live` and `vsss-sim`, hosted by the editor server.
Other VSSS tools can connect normally. Switching environment requires stopping
the student program first. A provider started externally is reused but never
terminated; stop it in its own terminal before switching Live/Simulation mode.

The built-in plot reuses `vsss.plot.EmbeddedPlot`, the BoardPlotter2D renderer,
ProviderFeed, and SimDrag. It has **Full screen** (the same button exits) and
**Collapse / Expand** controls.
Collapsing the field pauses plot requests. Cars 1–4 use the shared car sprites,
stretched to 15 × 15 cm squares in both this view and `vsss-plot`.
In simulation, drag car centres/balls to move and car
edges to rotate. Live is observation-only. Stale gestures are rejected after a
scenario/provider change; abandoned grabs expire. The server stops only providers
it owns on Ctrl+C. Set `VSSS_ENDPOINT` before starting to isolate a session.
Install the parent plot extra (`python -m pip install -e ".[plot]"`) for the view.

## Examples

Every requested example is available in the example selector and as a Python file.
Choose **Blank workspace** to clear the editor and start from scratch:

The dropdown also includes **Kick ball to top wall** and **Goalie: follow the
ball**. Kick reads the first ball, moves 30 cm behind it facing 90°, waits for
arrival, then drives forward for .8 seconds and stops. Use the **1 car 1 ball**
environment with a stationary ball and a clear approach; leave room behind it.
Goalie follows the ball's current X, reading positions every .05 seconds. With
the car facing 90°, sideways speed is `(car X - ball X) * response`, limited to
-1..1. It stops if no ball is visible. The default `response = 2` saves easy but
misses hard; try `response = 8` for a quicker response using the same rule.
There is no prediction or intercept algorithm.
These examples deliberately use ordinary editable blocks and assume one ball.

Both goalie scenarios place the car at Y=-1.2 m, facing 90°, with the ball above
at Y=.8 m. Easy starts the car at X=.15 m and the ball at X=.3 m with velocity
(-.1, -.6) m/s. Hard starts the car at X=-.3 m and the ball at X=.3 m with velocity
(-.3, -1) m/s. Both keep the ball paused until the first movement
command so students can prepare the program before the shot starts.

The same two programs are available as
[08_kick_top_wall.py](examples/08_kick_top_wall.py) and
[09_goalie.py](examples/09_goalie.py).

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

The rotation and waypoint examples use the selected environment. Live can move a
physical car; simulator scenarios control the simulated car.
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
.\.venv\Scripts\python.exe python_blockly\examples\03_rotate_back_and_forth.py
```

For standalone robot scripts, VSSS must be installed in that environment
(`pip install -e .`). The two basic Python scripts need no VSSS dependency.
Other examples retained under More examples cover two cars, two gamepads,
keyboard control and field observation.

## Student toolbox

Core blocks cover selecting/naming cars, velocity, waypoints, stop, position and
heading, other cars and balls, keyboard/gamepad held, pressed, released and
toggled states, normalized axes, print, variables, arithmetic, conditions,
loops, waits and lists.

Multiple named cars can be commanded in one loop or placed in a list for a
for-each loop. Commands are sequential, while each car independently continues
its most recent command. Velocity uses BLE only. Field reads start GPS lazily;
waypoint control needs the robot's separate onboard position reception.

**Advanced lessons** groups clock/timeouts, waiting for camera position and
readable robot checks:

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

Input reads use `vsss.input` on the server computer. The keyboard block offers
letters, digits, Space, Enter, Shift and the four arrow keys; browser/system
function keys are deliberately omitted. It can read held, just-pressed,
just-released or toggled state. Gamepads are numbered from 0; helper blocks
generate names such as `LX0`, `LY1`, `A1` and `L31`. The input parser currently
supports device suffixes 0-9. The module's native watchers are loaded only on
the first input read and are reused for the server lifetime. No browser input
emulation is added.

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

Runtime tests use fake devices and actual input API functions with seeded state. They
check multi-car routing/cleanup, instructor-operation rejection, source-driven
schema, each Python example, opposite rotation commands, repeated waypoint targets,
position-only observation, generated blocks, export, save/load, Stop and page
closure. Physical Bluetooth, gamepads and motion still need a supervised lab check.
The environment tests and `smoke_environment.cjs` also use a real local simulator
and RPC, including browser canvas dragging and scenario reset. They never select
physical hardware.
Save blocks explicitly before closing; there is no autosave. This localhost runner
is intended for a trusted classroom computer, not untrusted-code hosting.
