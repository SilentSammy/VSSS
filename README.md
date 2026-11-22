# VSSS IPC experiment

This is the streamlined provider architecture promoted from `ipc_cleanup`.
One provider composes a `GPS` and a `CarCmd`. The same client reads positions,
sends commands, and assigns callbacks against live or simulated providers.
Playback supplies the same observations and rejects commands.

See [PARITY.md](PARITY.md) for the source-based comparison with `overhaul`,
implemented parity, design evidence, and the remaining lab acceptance checklist.

Python 3.10+ is required. From this checkout:

```powershell
python -m pip install -e ".[plot]"
python -m vsss.simulation examples/scenario.jsonl
```

Installing editable changes the environment's `vsss-*` entry points to this
checkout. To keep the old installation selected, use an isolated virtualenv;
the module commands below also run directly from this directory if Pyro5 and
Bleak are already installed. No camera/OpenCV, gamepad or firmware dependencies
are imported by simulation or playback.

## Student Blockly editor

```powershell
.\.venv\Scripts\python.exe python_blockly/server.py
```

Open http://127.0.0.1:8765. Choose Live or a simulator scenario from Environment;
the server starts the corresponding ordinary VSSS provider if needed. Scenario
changes reset an existing simulator without restarting it. Start/reset reloads
the selected scenario. Stop a running student program before switching modes.
The embedded plot shares `vsss-plot` rendering and supports car/ball dragging in
simulation. Providers started in another terminal are reused and left running;
stop them there before changing between Live and Simulation. See
[python_blockly/README.md](python_blockly/README.md) for examples and tests.

## Manual test: separate terminals

Open PowerShell terminals in this checkout, using the same Python environment.
Install the current command entry points once (repeat after switching branches):

```powershell
cd C:\Users\SAMMY_LOQ\Desktop\Python\VSSS
.\.venv\Scripts\python.exe -m pip install -e ".[plot]"
```

1. Terminal A: `vsss-sim`. Leave it running throughout the simulation tests.
2. Terminal B: `vsss-plot`. A field window shows car1 and ball0. Escape or the
   close button closes this client. Positions use meters, +x right and +y up.
3. Terminal C: `vsss-teleop --car 1`. The previous `input.py` keyboard/gamepad
   controls run globally at 50 Hz; there is no teleop window. Hold W/S to move
   forward/back, A/D to strafe left/right, Q/E to rotate. Release keys to stop.
   C boosts from half to full speed. Motion is relative to the car's heading.
   Ctrl+C in the terminal stops and exits. Keyboard controls remain active when
   the plot has focus, as in the previous tool.
4. Terminal D: `vsss-sim --load examples/goalie_easy.jsonl`. This connects to
   the existing simulator, replaces its world, clears commands and exits.
   The plot and teleop stay connected. Car1 now faces up near the bottom of
   the field. This example defines starting poses; it does not run a goalie
   strategy. Repeat with `vsss-sim --load examples/scenario.jsonl` to reset.
5. Terminal D: `vsss-record trial01.jsonl`. Use teleop to move/rotate car1 and
   optionally load another scenario while recording. In terminal D press Ctrl+C
   to finish the recording. Use a new filename for each take; overwrites fail.
6. Ctrl+C in teleop, then Ctrl+C in terminal A to stop the simulator.
   Run `vsss-playback trial01.jsonl --speed 0.5` in terminal A, then
   leave `vsss-plot` open in terminal B. The plot reconnects automatically.
   Playback waits for the plot's subscription and replays at half speed.
   It stays open at the final frame; Ctrl+C stops it.
   `--loop` repeats the recording. Teleop is rejected during playback.
7. For event text in another terminal: `vsss-observer`. It shows position,
   command and lifecycle events. Stop it with Ctrl+C.

Only one provider (sim/live/playback) runs on the default endpoint at a time.
Clients can run together. Plot can start before a provider and reconnects after
provider changes. Other clients must be restarted after a provider changes.
`vsss-sim examples/goalie_easy.jsonl` starts a new server with that initial
scenario; `vsss-sim --load ...` changes an existing server.

To try real hardware instead, stop playback/simulation and start
`vsss-live --vision-server 192.168.12.1`, then use the same plot, teleop,
record and observer commands. Teleop now moves the physical car; start with
`vsss-teleop --car 1 --speed 0.1`. The plot requires incoming vision telemetry.

The plot reuses `aruco_gps.plotter2d.BoardPlotter2D` and the local lab definitions
in `aruco_gps/vision_setup.py`. It uses `DEFAULT_SETUP` by default (currently
`setup_160cm`), including the board PNG artwork, marker dimensions, margins,
colors, labels and image sprites from the setup's `marker_config.json`.
The simulator's default field and plot mismatch check use the selected default
board's printable dimensions, including its margins (currently 1.6 x 3.0 m),
rather than the smaller ArUco grid dimensions. Select the same setup as the lab
server:

```powershell
vsss-plot
vsss-plot --list-setups
vsss-plot --setup setup_letter
vsss-plot --setup setup_90cm
vsss-plot --setup setup_160cm
vsss-plot --setup setup_90cm --marker-config my_markers.json
```

Setup and artwork are local files; they are not fetched through UDP, HTTP or
RPC. Keep the lab and plot checkout/config/assets synchronized manually.
Relative sprite paths resolve from the marker JSON's directory. Marker height
overrides belong to lab detection; plot only uses visual overrides. Restart
plot after editing the local config. Setup geometry stays fixed across provider
changes; selecting a setup does not resize the simulator or change the server.
A field-size mismatch appears in the plot status, but matching sizes do not
prove that artwork/settings match. PNG artwork and default marker config/sprites
are included in built wheels as well as usable directly from the checkout.

Install the plot extra and initialize the submodule (`git submodule update --init`)
on a fresh checkout. `vsss-plot --duration 2 --output field.png` saves a headless
frame to a new file. `--width 720` sets resolution; Escape closes the window.
The window close button and Alt+F4 also work normally. Q remains available to
teleop while the plot has focus.

Teleop retains `--list`, `--refresh`, `--auto`, repeated `--car ID[=INPUT+INPUT]`
and interactive assignments. Examples:

```powershell
vsss-teleop --list --refresh
vsss-teleop --car 1=wasd+gamepad:0 --car 2=ijkl
vsss-teleop --auto
```

I/K, J/L, U/O and M are the second keyboard layout. Gamepad left stick
translates, right stick rotates, and RT boosts. `--speed 0.3` multiplies both
normal and boosted speeds. Interactive startup (`vsss-teleop` with no selection
flags) always forces a new scan. `--auto` and `--list` may use the 30-second
discovery cache unless `--refresh` is supplied. Explicit `--car` selections keep
scanning until each car connects, and retry after disconnection without stopping
the other cars. Release controls to neutral after reconnection before moving;
queued motion is discarded. Bare `--car 1` combines WASD and gamepad 0, including
a pad plugged in after startup; `--car 1=wasd` selects keyboard only. Additional
bare car selections use the next unused keyboard and gamepad, when available.
Only changed, quantized inputs send commands. A background writer
keeps at most one pending command per car while the input loop continues at
50 Hz, even when the BLE acknowledged write is slow. Newer input replaces
unsent intermediate input. Shutdown discards pending motion, finishes the
in-flight call, and then sends an acknowledged STOP.

To compare driving responsiveness with IPC removed, stop teleop and `vsss-live`
first to release the BLE connection, then run in the same activated environment:

```powershell
python examples/teleop_direct.py --car 1
```

This instantiates `VSSSProvider(GPSLive(...), CarCmdLive(...))` in the teleop
process, without hosting or connecting to RPC. It uses the exact same input
loop, latest-command writer, speed flags, BLE adapter and vision defaults as
the RPC setup. Ctrl+C stops and closes the local provider. For comparison, exit
the direct script, restart `vsss-live`, and run `vsss-teleop --car 1`. Use the
same controls and speed in both runs. Direct mode is not visible to RPC plot or
record clients because it does not publish an endpoint.

## One client API

```python
from threading import Event
from vsss import VSSSProvider, Velocity, StatePatch, CarState, Pose

with VSSSProvider.connect() as provider:
    print(provider.describe())
    print(provider.gps.state())  # WorldState, containing typed poses and field
    print(provider.gps.age())    # seconds since last delivered observation
    provider.gps.on_change = lambda state: print(state.cars)
    provider.cars.on_change = lambda command: print(command)
    provider.cars.set_velocity(1, Velocity(x=0.2))
    provider.cars.stop(1)
    # Simulation only; moves this car and stops its active command:
    provider.update_state(StatePatch(cars=[CarState(1, Pose(.1, .2))]))
    # Keep the process alive to receive callbacks:
    Event().wait()
```

Server/client methods use the same records and signatures. Components contain
normal methods marked `@rpc_method`; a generic client/dispatcher forwards them.
`@record` dataclasses automatically encode nested records, lists, maps and
primitive values. Unregistered types and nonfinite floats are rejected.
No dynamic imports or arbitrary class construction occur during decoding.

`gps.on_change` receives `WorldState`; `cars.on_change` receives `CommandEvent`.
Callbacks run on background threads and may issue commands. They should remain
short. For complete event envelopes or an inspectable failure handle:

```python
with provider.subscribe(["gps", "commands", "lifecycle"], print) as subscription:
    # subscription.error reports callback, disconnect and overflow failures.
    ...
```

Every subscription has a bounded queue. A slow subscriber fails explicitly on
overflow; it cannot block telemetry/control producers. Client connection EOF
stops cars owned by that session. Reads do not take ownership; the latest
successful explicit command does. There is no automatic reconnect or replay
of uncertain commands.

STOP clears motion ownership. Device I/O and ownership changes serialize per
car; a blocked radio does not hold a fleet-wide lock. Teleop has an independent
latest-value worker and RPC connection per assigned car. `client.fork()` creates
another generic session to the same provider when parallel requests are needed.

## Waypoints, feedback and student helpers

```python
from vsss import VSSSProvider, Pose

with VSSSProvider.connect() as provider:
    provider.cars.connect(1)  # Stopped setup; does not release a waiting sim.
    provider.cars.go_to(1, Pose(0.2, 0.2, 90))
    status = provider.cars.status(1)  # Typed RobotStatus
    print(status.mode_name, status.fault_name, status.pose_fresh)
    provider.cars.stop(1)
```

Waypoints use field metres and CCW degrees. Velocity takes over from navigation.
`go_to` acknowledges acceptance; poll status for arrival, as in the autonomous
boilerplate.
STOP, disconnect and edits cancel active and delayed navigation. Live waypoint
heartbeats use the firmware lease characteristic instead of velocity. Simulation
runs `mp.onboard_controller` with arrival holding, recovery and timeouts.

`gps.state()` contains only usable objects; stale, board-lost and missing objects
are omitted. Live ingress rejects excessive camera age and reordered/delayed
packets. `gps.get_car(ID)` returns a Pose or None. `gps.age()` incorporates camera
frame age, excess network delay and time since delivery. `gps.snapshot()` is an
explicit historical view. Source `captured_at` and delivery `received_at` remain
distinct; playback preserves source time and uses fresh delivery timing.

The student helpers delegate to the same methods:

```python
from vsss import Car, gps, run

@run
def main():
    car = Car(1)
    car.wait_for_position()
    car.go_to(0.2, 0.2, 90)
    print(car.read_status().mode_name)
    # @run stops/closes cars on return, exception or Ctrl+C.

if __name__ == '__main__':
    main()
```

`GPS` remains the abstract component; `ClientGPS` is the student view and `gps`
is lazy. Velocity-only code does not start GPS. `Car(1, provider=local_provider)`
works without RPC. The restored `boilerplate_velocity.py`,
`boilerplate_observe.py`, `boilerplate_autonomous.py` and `goalie.py`, as well as
Blockly's live examples, use these helpers. Start a provider separately. Set
`VSSS_ENDPOINT` for scripts/Blockly when using a custom provider `--endpoint`.

Admin methods are `read_trim`/`set_trim`, `read_controller`/`set_controller`,
`read_robot_config`/`set_robot_config`, `reconnect_telemetry`, and `restart`, each
taking a car ID. Mutations stop first; live writes validate/read back canonical
values and chunk full configuration over BLE. Nothing resumes previous motion.
Configuration results may contain credentials and are returned only to the
explicit caller, never recorded in events. Simulation supplies the same methods;
its identities come from the scenario. `Car.restart()` waits for a stopped
reconnect; the component call acknowledges the restart and later commands
reconnect explicitly/lazily.

## Providers

```powershell
python -m vsss.simulation
python -m vsss.simulation examples/scenario.jsonl
python -m vsss.live --vision-server 192.168.12.1
python -m vsss.playback examples/recording.jsonl --speed 2
```

Only one provider binds the default port 38475. It publishes a loopback Pyro URI
and random handshake token in a per-user temporary discovery file. Use
`--endpoint PATH --port N` to isolate an experiment; clients pass the same path
to `VSSSProvider.connect(PATH)`. There is no detached auto-start yet.

Playback waits for its first successful event subscription by default. Assigning
a callback establishes that subscription. Use `--immediate`, `--loop`, or the
RPC `start_playback()` for explicit alternatives. Completion is a lifecycle event;
the provider remains available until Ctrl+C for inspecting final state.

The live GPS adapter receives existing vision datagrams and fetches field
dimensions once over HTTP; `gps.refresh_field()` retries on demand. Its initial
state is invalid until telemetry arrives. The BLE adapter lazily connects to
`carN`, sends an initial stop, acknowledges velocity writes, maintains independent
heartbeats and stops/disconnects during shutdown. `cars.status(N)` reads onboard
status. BLE UUIDs, payloads and UDP packets stay inside `live.py`.

The simulator publishes at 30 Hz by default and advances Pymunk physics at a
fixed 240 Hz. Square car bodies have finite motor acceleration; cars contact
one another, push balls, and collide with field walls. Balls bounce and lose
speed through rolling resistance. Full speed is 0.6 m/s and 180 degrees/s.
Defaults are illustrative. `--config` loads physical/motor/wobble/delay settings;
`--ball-model` accepts model JSON or a fit report containing `ball_model`/`model`.
`examples/physics_lab.json` ports the former fitted impact/rolling/drift values
and delays. Ball contacts preserve the former normal impact-gain semantics while
Pymunk resolves geometry/tangential contact; hardware fidelity still needs lab
verification.
Editing an object clears its physical momentum; loading a scenario resets the
whole physics world. Unedited objects retain motion.

`load_scenario(WorldState)` replaces the complete world and clears commands.
`update_state(StatePatch)` changes only listed car/ball poses; it validates the
whole edit before applying it and stops edited cars. It also accepts a JSON
string containing the tagged patch. Scenario files contain a tagged WorldState
on their first line, or a new-format recording header. A scenario is an initial
state, not a mid-run restoration of timers/random generators.

`SimScenario` also holds `BodyMotion` initial velocities (world m/s, degrees/s),
config, seed and a control-start gate. Two-frame `sim_scenario` JSONL files derive
motion from increasing times and matching IDs, including wrapped headings.

```powershell
vsss-sim --scenario goalie --seed 123 --config examples/physics_lab.json --save-scenario shot.jsonl
vsss-plot
python goalie.py --duration 5
vsss-sim --load shot.jsonl
```

Use separate terminals while sim stays open. The goalie scenario waits for an
explicit velocity/waypoint before physics starts; connection, reads and admin
setup do not release it. `--wait-for-control` enables this for other scenarios.
Simulated BLE delay affects incoming motion; camera delay affects observations
and onboard feedback. STOP cancels delayed motion immediately. `scenario()`
exports current poses/motion/config as an initial state, not a controller/random
stream checkpoint.

The goalie example predicts a crossing, holds its original Y lane and corrects
small heading disturbances caused by dynamic contacts. It stops on feedback loss
or excessive lane/heading departure. Five seeded simulator shots are regression
cases; they do not establish a physical lab success rate.

Plot supports dragging car centres/balls to move and car edges to rotate. Grabs
anchor bodies and reject other editors/commands until release. The worker
coalesces moves off the GUI thread, preserving grab/release order. Closing or
changing providers releases grabs. Source IDs and per-grab tokens reject stale
gestures after a provider/scenario change. Scripted patches remain atomic.

## Record and replay

Installed commands include `vsss-live`, `vsss-sim`, `vsss-plot`, `vsss-teleop`,
`vsss-playback` (also `vsss-bag`), `vsss-observer`, and `vsss-record`:

```powershell
vsss-record experiment.jsonl
vsss-observer
vsss-playback experiment.jsonl --speed 0.5
```

Recordings use `vsss.recording/v1`: one initial state/command/status snapshot, timed
domain events, and a clean/incomplete end marker. Topics are `gps`, `commands`
and `lifecycle`; the format has no UDP/BLE UUIDs, packets or base64 fields.
Legacy schema-3 bags are intentionally rejected. See `examples/recording.jsonl`.

Velocity/waypoint commands and observed onboard statuses replay as typed domain
values. A bag with no recorded status returns unknown status rather than
fabricated fresh onboard feedback. Admin arguments/results are not recorded.

## Verification and rollback

```powershell
python -m unittest discover -s .tests -v
python -m unittest discover -s python_blockly -p test_live_runtime.py -v
python .tests/hardware_smoke.py --car 1
python .tests/latency_probe.py --samples 300
```

The normal tests use synthetic UDP, fake BLE and real subprocess RPC; they do
not move hardware. The explicit hardware check scans, reads status and sends
only STOP. It is not part of automatic test discovery.

The latency probe normally measures acknowledged command RPCs to a separate
local process with no hardware. Add `--stop-only --car 1` to compare those with
the running provider's read-only RPCs and actual BLE STOP completions. This
explicit mode refuses to run while a nonzero command is observed. These timings
measure call completion, not the instant of motor response. Live heartbeats
are sent only after 200 ms without a successful velocity write; active commands
renew the lease themselves, so they do not require duplicate heartbeat writes.

The pre-cleanup package and docs remain in Git at commit `8a58b90`.
Obsolete tools/tests/scripts, old scenarios and generated artifacts were moved
outside the checkout to `../ipc_sandbox/vsss-legacy-before-cleanup-dc7351f/`.
This includes ignored recordings and hardware backup files. Firmware under
`mp/` and the vision submodule remain in place. The cleanup is now the `overhaul`
implementation after parity verification. Selected boilerplates/goalie were
restored on top of the new provider; the old
provider/worker IPC stack remains removed. Restart provider/client processes
after updating so both sides load the new method and record definitions.
See `DEV.md` for module responsibilities and current limitations.
