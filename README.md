# VSSS

Python control and field observation for the lab's mecanum cars. Enable Bluetooth
and join the lab network (`embedded`) to use the real cars and vision feed.
Implementation and maintenance notes are in [DEV.md](DEV.md).

This project is the software platform for a small fleet of ESP32-based mecanum
robots used in the VSSS lab. Students can write Python programs that drive cars,
observe their positions and the ball field, and combine those capabilities for
autonomous behavior. The cars receive commands over BLE, while the shared vision
system publishes marker and ball telemetry over UDP. The repository also includes
provider, replay, plotting, teleoperation, calibration, and development tools.

## Installation

Install Git and Python 3.10 or newer, then clone into any folder:

```sh
git clone --recurse-submodules https://github.com/SilentSammy/VSSS.git my-vsss
cd my-vsss
```

The checkout name is arbitrary. Run the following commands from its root; no
machine-specific paths are needed. If you move the checkout later, rerun setup
to recreate its virtual environment.

Windows (PowerShell):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
./setup.ps1 -Yes
. ./.venv/Scripts/Activate.ps1
python -m pip install -e ".[plot]"
```

Linux:

```sh
bash setup.sh --yes
source .venv/bin/activate
python -m pip install -e ".[plot]"
```

Activate `.venv` again in each new terminal. The vision server runs separately
on the lab computer; students do not need to install or start it.

## Commands

Run any implemented command with `-h` for its arguments.

| Command | Purpose |
| --- | --- |
| `vsss-live` | Run the real BLE/UDP provider until Ctrl+C. Starts automatically when needed if no provider is running. |
| `vsss-bag FILE` | Serve a recorded bag. Waits for the first feed subscriber before playback (`--immediate` skips waiting); exits when playback ends (`--loop` repeats until Ctrl+C). |
| `vsss-sim` | Serve simulated cars and field telemetry until Ctrl+C; existing car scripts and tools work unchanged. |
| `vsss-observer [FILE]` | Print UDP/BLE activity, or record it to a new bag file with the cached vision setup header. |
| `vsss-plot` | Plot the active provider's field telemetry using its setup metadata. `--persist` waits and reconnects across feed endings and provider restarts. |
| `vsss-teleop` | Discover cars and assign keyboard/gamepad controls. Sends velocity commands only; explicit arguments allow promptless use. |
| `vsss-dance` | Fleet motor/BLE check: alternate gentle rotation and continuously discover new cars. Ctrl+C stops the fleet. Use `--car 1 --car 2` to restrict selection, `--duration 15` for a timed run, or `--speed .2 --interval 1` to adjust motion. Does not verify GPS/UDP feedback. |
| `vsss-waypoint` | Planned: send waypoint commands from a terminal. |
| `calibrate --car ID` | Measure and propose trim; optionally save/apply it, apply a saved calibration, or reset trim. Measurement mode moves the car. |

Only one provider mode can run at a time. Start `vsss-bag` before your consumers
to use playback; it exits at the end, or can be stopped with Ctrl+C before switching modes. An automatically
started live provider exits five seconds after its last client leaves.

Teleop's keyboard aliases are `wasd` (Q/E rotation, C boost) and `ijkl`
(U/O rotation, M boost). Examples:

```sh
vsss-teleop --car 1=wasd --car 2=gamepad:0
vsss-observer session.jsonl
```

By default, `vsss-plot` exits when playback ends or its provider disconnects.
Use `vsss-plot --persist` to leave the window open while switching providers.
This mode waits for an existing provider, including at startup, without starting
live mode automatically. Start `vsss-live` or `vsss-bag FILE` separately; each
connection reloads the field setup. Press Q/Escape or close the window to exit.

## Simulation

Start the simulator before starting clients. It takes the same provider slot as
live hardware or bag playback, without using Bluetooth, Wi-Fi, or the camera:

```sh
vsss-sim
```

To start from a recording, use `vsss-sim session.jsonl`. The header supplies field
metadata and the first valid frame with a detected board supplies initial car
and ball poses. Objects start at rest. Older bags without setup metadata use
default field settings; `--config` can override them. The simulator uses just
this initial frame and then evolves normally.

In separate terminals with `.venv` activated, run the usual tools:

```sh
vsss-teleop --car 1=wasd
vsss-plot --persist
vsss-observer simulation.jsonl
```

On a simulator feed, the plot shows each car's body outline and a center ring.
Drag inside the ring to move the car, or drag its outer body to rotate it in
place. Drag a ball to move it. A dragged object stays held until mouse release;
releasing it leaves it stopped. Movement is clamped to the field. Live and bag
playback feeds remain view-only. `--persist` switches editing on or off as the
active provider changes.

Other processes can reposition objects through the same local simulator server:

```python
from vsss.sim import SimClient

with SimClient() as sim:
    sim.move_car(1, x=0.2, y=-0.4, angle=90)
    sim.move_ball(0, x=0.0, y=0.3)  # balls use zero-based indices
```

Use `sim.grab('car', 1)` or `sim.grab('ball', 0)` to hold an object against
physics while moving it repeatedly, then `sim.release(kind, id)` to let go.
Closing the client also releases its grabs. Moving a car cancels its current
motion/waypoint; a new control command can move it again after release. The
editing client never starts a provider or sends commands to real cars.

The default world has car 1 facing the ball, a 1.6 by 3 metre enclosed field,
and a 14 cm square car body. `Car(1)`, GPS, waypoints, trim/controller readback,
shared motion ownership, and stop-on-owner-exit use the normal API. Recordings
use the current bag format and can be replayed with `vsss-bag`.

Use `vsss-sim --cars 1 2` for a two-car fleet. `--config FILE.json` overrides
settings from [sim_defaults.json](vsss/sim_defaults.json), including initial
poses, balls, field dimensions, motion response, physics/telemetry rates, and the
car's convex body vertices in local metres (counterclockwise around the origin).
`--ball-model FILE.json` accepts a ball model or `tools.ball_dynamics` fit report.
Settings last for the provider process; a new process starts a fresh world.
Initialization order is bundled defaults, bag/scenario setup, JSON config, then
`--ball-model`, `--cars`, and `--seed` overrides.

The ball uses the measured impact and rolling-resistance fit. Car speed/response,
wall bounce, and car-to-car constraints are approximations; this is an initial
algorithm-testing simulator. Model evidence and limitations are in [DEV.md](DEV.md).

Slow balls also wobble and vary their braking as they settle, based on terminal
motion in the lab recordings. Each run uses a fresh random seed, printed at
startup and saved in bag metadata. Use `vsss-sim --seed 123` for repeatable motion.
The `ball_wobble` settings in the JSON config tune the strength and timing;
`{"ball_wobble": {"enabled": false}}` restores the original smooth ball model.
Wobble changes physical motion; it does not make resting balls or camera
observations jitter.

The simulator also adds simple fixed transmission delays: 100 ms for UDP poses
and 40 ms per BLE request. The onboard waypoint controller receives delayed
poses too. Change them with a JSON config, for example
`{"latency": {"udp_ms": 150, "ble_ms": 60}}`; set both to zero to disable them.

## Scenarios and a simple goalie

Start a random incoming shot, then run the goalie in another terminal:

```sh
vsss-sim --scenario goalie
python goalie.py
```

The ball starts at a random position with `y <= 0`, aimed at a random point
on the positive-Y edge of a 1.5 by 2.9 m field. Car 1 starts at `(0, 1.20)`,
facing +X. The scenario waits for the first car velocity/waypoint command, so
opening `vsss-plot` or connecting a car does not consume the shot. The goalie
starts it with an explicit zero command. Restart the simulator for another
shot; add `--seed 42` to repeat the same start and wobble.

[goalie.py](goalie.py) uses only `Car`, `gps` and `run`. From two ball observations
it projects the crossing at its fixed Y: `x_hit = x + vx/vy * (goalie_y - y)`.
A small proportional controller slides toward that X, bounded to the field.
Every command has `y=0` and `w=0`. This happy-path lesson assumes one straight
incoming shot, no wall bounce and a car facing +X; it stops on missing feedback
or leaving its horizontal lane. Use `--duration 8` for a finite run, or Ctrl+C.

Scenarios encode initial velocity using **two timed position lines**:

```jsonl
{"type":"sim_scenario","schema":1,"config":{"field":[1.5,2.9],"wait_for_control":true}}
{"time":0.0,"cars":[{"id":1,"x":0,"y":1.2,"angle":0}],"balls":[{"x":-0.25,"y":-0.7}]}
{"time":0.1,"cars":[{"id":1,"x":0,"y":1.2,"angle":0}],"balls":[{"x":-0.23,"y":-0.62}]}
```

The simulator starts at the **first** line's positions, with velocity
`(position2-position1)/(time2-time1)`. Here the ball starts at `(-0.25, -0.7)`
with `(vx, vy) = (0.2, 0.8)` m/s. Match cars by ID and keep balls in the same
order on both lines. Times are seconds, positions metres and car angles degrees.
[scenarios/goalie.jsonl](scenarios/goalie.jsonl) is a ready-to-edit example.

```sh
vsss-sim --scenario scenarios/goalie.jsonl
vsss-sim --scenario goalie --save-scenario my-shot.jsonl
vsss-sim --scenario my-shot.jsonl
```

`--save-scenario` writes the resolved start to a new file, including its seed,
then runs it. Each command above starts its own simulator; stop the current one
before starting another. Ordinary `vsss-sim BAG` still loads one frame at rest.
Use `vsss-sim --scenario BAG` to infer motion from its first two usable field
frames instead. Bag ball matching is by list order; choose a bag segment with
stable detections. A scenario header can override normal world settings;
`--config` and other explicit overrides still apply afterward.

## Student example: move a ball into a zone

Start `vsss-sim --seed 123`, then in another terminal run:

```sh
python ball_to_zone.py --car 1 --target 0 1 --radius 0.18
```

[ball_to_zone.py](ball_to_zone.py) uses only the student-facing `Car`, `gps` and
`run` API. Its strategy is:

1. Find the direction from the ball to the goal center.
2. Use waypoints to stand 23 cm behind the ball, going around it if needed.
3. Capture at up to 0.30, then push at a normalized cruise command of 0.40
   with small sideways corrections. Lock rotation once contact is observed.
   Taper to 0.30 near the goal and re-approach if contact/alignment is lost.
4. Release near the zone entrance, allowing for the final coast, and check
   that the ball settles there. If it escapes, wait for it to stop before
   planning another approach.

This is a continuous-push example rather than a prediction of a perfect kick.
It assumes one visible ball starting at rest, a clear field, the 14 cm car and
enough room to get behind the ball. It stops on missing feedback and has a time limit. It
does not change trim/controller settings or access simulator state. The same
script can drive a real car; the reported trials were performed in simulation.
The example defaults to the usable 1.5 by 2.9 metre field recorded in the live
bags; use `--field WIDTH HEIGHT` for another size. It checks observed positions
and refuses outward pushes near an edge. Stopping the car cannot arrest a
ball already coasting outside the field.
`--push-speed 0.4` sets the cruise command (supported range 0.25-0.6). Review
of bag 12 showed the earlier 0.35 revision repeatedly switching between forward
motion and centering/turning, often at just 0.20-0.25. The current controller
uses continuous forward feed, limits yaw before contact, and keeps yaw at zero
for the rest of the push once contact is seen. Large alignment errors require
a new waypoint approach. This revision still needs a fresh physical trial.
For a first lesson, use the default simulator layout: the car starts behind
the ball, so the approach needs just one waypoint. Going around the ball can
be taught as a later extension.

For a repeating demonstration, run `python ball_back_and_forth.py`. It alternates
between 18 cm zones centered at `(0, 1.0)` and `(0, -1.0)`, pausing one second
after each arrival. Press Ctrl+C to stop and release the car. Use
`python ball_back_and_forth.py --cycles 1` for a single round trip.
The repeating demo accepts `--push-speed` and `--field` too.

For a simpler task, kick toward a point without trying to stop the ball there:

```sh
python kick_ball.py --target 0 1
python kick_ball_simple.py --target 0 1
```

[kick_ball_simple.py](kick_ball_simple.py) is the happy-path lesson: start with
a clear straight path to the point 30 cm behind the stationary ball. Compute
the unit direction `u = (target - ball) / distance`, go to `ball - 0.30*u`
facing `u`, then drive straight forward at full command. There is no steering
during the kick and no check that the ball stops at the target.

[kick_ball.py](kick_ball.py) adds a small detour when the ball blocks that
approach: sideways to a parallel line 30 cm from the ball, backward along it,
then inward onto the shot line. It needs at most three approach waypoints,
with no arc angles or repeated replanning. Both versions share observation,
waypoint, and strike helpers so the happy-path algorithm stays short.

The strike is `set_velocity(x=1, y=0, w=0)` until the observed car center passes
the ball's original position by 2 cm, then the **car** stops. A 1.5 second
limit and field/feedback checks bound the run-up. The ball is allowed to keep
rolling past the aim point; it is never chased. Assume one stationary ball,
a clear field and room for the run-up/follow-through. Both accept `--car` and
`--field WIDTH HEIGHT` (default 1.5 by 2.9 m). Practice with `vsss-sim` first;
the reported checks used simulation, not physical hardware.

## Python API

The public modules under `vsss/` are:

| File | Common use |
| --- | --- |
| [car.py](vsss/car.py) | `Car(ID)` connects to a numbered car over BLE. |
| [gps.py](vsss/gps.py) | `gps` lazily starts shared field observation; `GPS()` accesses the same receiver. |
| [input.py](vsss/input.py) | Read keyboard and gamepad state in custom scripts. |
| [provider.py](vsss/provider.py) | `available_cars()` returns car IDs; `ProviderClient` exposes raw observation events for tools. |
| [__init__.py](vsss/__init__.py) | Convenient imports: `Car`, `GPS`, `gps`, `Pose`, `Position`, and `run`. |

### Car

- `set_velocity(x=0, y=0, w=0)`: normalized forward, left, and counterclockwise
  commands in `[-1, 1]`; `stop()` sends zero.
- `go_to(x, y, angle)`: start onboard waypoint navigation using field coordinates.
  Requires healthy telemetry and the car's own marker to be visible.
- `pose` / `position`: current field pose or XY tuple; `None` when unavailable.
  `wait_for_position()` waits for a valid pose.
- `read_status()`: onboard mode, fault, pose, and telemetry counters.
- `read_trim()` / `set_trim(matrix)`: inspect or replace the correction matrix.
  `read_controller()` / `set_controller(parameters)`: inspect or update controller settings.
- `read_robot_config()` / `set_robot_config(config)`: read or replace the complete
  persistent `robot_config.json`, including car identity, Wi-Fi/telemetry, motor
  pin pairs, trim, and controller settings. A replacement stops the car and
  reboots it after saving, so reconnect using its new ID if `car_id` changed.
- `is_connected`: BLE readiness. `close()` / `Car.close_all()`: release this
  script's connections. Prefer `@run` for automatic cleanup.

Velocity control needs only BLE; GPS starts when position is requested.
Multiple scripts can share a car. The latest explicit motion command takes
control, and velocity cancels a waypoint. The motion owner leaving stops the
car; another client's departure does not interrupt it.

Motor pin mappings are four `[forward_pwm, reverse_pwm]` pairs ordered
front-left, front-right, back-left, back-right. They live in the per-car
`robot_config.json`, defaulting to `[[3,4],[1,0],[7,5],[8,9]]`. Pin changes,
as well as identity and network changes, take effect after the automatic
restart. For example:

```python
config = car.read_robot_config()
config["car_id"] = 3
config["motor_pins"] = [[3, 4], [1, 0], [7, 5], [8, 9]]
car.set_robot_config(config)
```

### GPS and input

`gps.cars` maps marker IDs to `Pose(x, y, angle)`; `gps.get_car(ID)` returns
one pose or `None`. `gps.balls` contains `Position(x, y)` objects.
`gps.board_detected` indicates board visibility. Stale field observations
return empty collections. Field distances are metres and headings are degrees,
counterclockwise from +X. GPS observation does not require a BLE connection.

For custom input, use `input.is_pressed(key)`, `rising_edge(key)`,
`falling_edge(key)`, and `get_axis(name)`. Gamepad names include `LX`, `LY`,
`RX`, and `RT`; an index suffix selects a device, e.g. `LX1`.

```python
from time import sleep
from vsss import Car, run

@run
def main():
    car = Car(1)
    car.set_velocity(x=0.2)
    sleep(1)
    car.stop()

main()
```

`@run` releases the script's cars and GPS on return, error, or Ctrl+C.
Starting scripts: [velocity](boilerplate_velocity.py),
[field observation](boilerplate_observe.py), and
[autonomous control](boilerplate_autonomous.py).

## Blockly

## Staged axis and mixed-axis calibration

`tools.calibrate_staged` fits single-axis trim, verifies it with fresh pulses,
then fits and verifies the unwanted third-axis motion during dual-axis commands.
Each stage records its starting pose and observes short BLE velocity bursts
directly through host UDP. Other marker detections, field boundaries and the
robot's onboard pose freshness do not gate this calibration. Between pulse
pairs, short single-axis correction bursts return toward the starting pose.
Each correction lasts 0.30 seconds for translation or 0.45 seconds for yaw,
followed by a settled UDP observation.

From the project virtual environment, run these stages in order, using a new
output directory for each complete calibration attempt:

```powershell
python -m tools.calibrate_staged --car 1 --output tools/runs/car1_staged --stage axes --execute --apply
python -m tools.calibrate_staged --car 1 --output tools/runs/car1_staged --stage axes-check --execute
python -m tools.calibrate_staged --car 1 --output tools/runs/car1_staged --stage mixed --execute --apply
python -m tools.calibrate_staged --car 1 --output tools/runs/car1_staged --stage mixed-check --execute
```

Without `--execute`, the command only previews. The two fitting stages need
`--apply` to write a passing proposal. Verification requires matching configuration
readback and fresh measurements. Failed verification restores the previous trim;
a link failure can prevent rollback. Each completed pulse, initial configuration,
proposal and verification result is recorded. Interrupted measurement stages
retain their exact configuration and samples; add `--resume` to the same command
to continue. `--retry` records a fresh verification attempt separately. A completed
verification that fails the measurement limits rolls back the candidate.
`tools.recheck_mixed` can reuse a complete verification and repeat only cases
above tolerance twice. It retains every raw observation and accepts a repeated
case only when the median absolute residual of all three observations meets the
same original tolerance; it does not widen the tolerance.
Use its `--axes` option for single-axis verification. When a full model update
overcorrects a nonlinear response, `--step`, `--yaw-steps` (forward and strafe),
and `--mixed-step` allow independently damped candidates from the same saved
measurements; every candidate still requires independent verification.

New staged runs default to `--burst-scale 1.6`: forward 1.28 s, strafe/yaw
1.92 s and mixed-axis 1.12 s. `plan.json` records the scale; resume rejects
changing it mid-stage. Legacy checkpoints keep their original duration.
For a remaining forward/reverse yaw bias, `tools.refine_forward_yaw` probes
only `trim_matrix[2][0]` using longer +/-x bursts, measures the local sensitivity,
fits a bounded secant update and independently verifies steady yaw rate. It
preserves all other axis/mixed coefficients and supports `--resume`.

`mixed_trim` in `robot_config.json` holds a row-major 3x6 matrix, with output rows
`x, y, w` and columns `x*abs(y), y*abs(x), x*abs(w), w*abs(x), y*abs(w), w*abs(y)`.
The firmware adds the resulting correction to the requested command before
applying the existing 3x3 `trim_matrix`. The mixed terms are exactly zero for
every single-axis command; zero coefficients preserve the previous behavior.
Each additional command is limited to +/-0.15. Fits require all four sign
combinations for every pair at two levels and reject excessive coefficients.
Behavior outside the measured range, particularly near motor deadband or wheel
saturation, still requires separate validation. The original variable-frequency
PWM remains enabled; calibration does not replace that low-speed feature.
