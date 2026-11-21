# VSSS IPC experiment

This is the `ipc_cleanup` branch: a deliberately incompatible, minimal rewrite.
One provider composes a `GPS` and a `CarCmd`. The same client reads positions,
sends commands, and assigns callbacks against live or simulated providers.
Playback supplies the same observations and rejects commands.

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

The plot reuses `aruco_gps.plotter2d.BoardPlotter2D`, the 3x5 ArUco board artwork,
pastel marker labels and orientation indicators, with no vision metadata request.
Install the plot extra and initialize the submodule (`git submodule update --init`)
on a fresh checkout. `vsss-plot --duration 2 --output field.png` saves a headless
frame to a new file. `--width 720` sets resolution; Q/Escape closes the window.

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
discovery cache unless `--refresh` is supplied; explicit car assignments bypass
discovery. Only changed, quantized inputs send commands. A background writer
keeps at most one pending command per car while the input loop continues at
50 Hz, even when the BLE acknowledged write is slow. Newer input replaces
unsent intermediate input. Shutdown discards pending motion, finishes the
in-flight call, and then sends an acknowledged STOP.

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
The masses, friction and motor constants are illustrative rather than measured;
firmware waypoints, noise and camera latency are still outside this core.
Editing an object clears its physical momentum; loading a scenario resets the
whole physics world. Unedited objects retain motion.

`load_scenario(WorldState)` replaces the complete world and clears commands.
`update_state(StatePatch)` changes only listed car/ball poses; it validates the
whole edit before applying it and stops edited cars. It also accepts a JSON
string containing the tagged patch. Scenario files contain a tagged WorldState
on their first line, or a new-format recording header. A scenario is an initial
state, not a mid-run restoration of timers/random generators.

## Record and replay

Installed commands include `vsss-live`, `vsss-sim`, `vsss-plot`, `vsss-teleop`,
`vsss-playback` (also `vsss-bag`), `vsss-observer`, and `vsss-record`:

```powershell
vsss-record experiment.jsonl
vsss-observer
vsss-playback experiment.jsonl --speed 0.5
```

Recordings use `vsss.recording/v1`: one initial state/command snapshot, timed
domain events, and a clean/incomplete end marker. Topics are `gps`, `commands`
and `lifecycle`; the format has no UDP/BLE UUIDs, packets or base64 fields.
Legacy schema-3 bags are intentionally rejected. See `examples/recording.jsonl`.

## Verification and rollback

```powershell
python -m unittest discover -s tests -v
python tests/hardware_smoke.py --car 1
python tests/latency_probe.py --samples 300
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

The original package and docs remain in Git (`overhaul`, commit `8a58b90`).
Obsolete tools/tests/scripts, old scenarios and generated artifacts were moved
outside the checkout to `../ipc_sandbox/vsss-legacy-before-cleanup-dc7351f/`.
This includes ignored recordings and hardware backup files. Firmware under
`mp/` and the vision submodule remain in place. No merge into `overhaul` has
been made; this branch is still being verified against the previous behavior.
See `DEV.md` for module responsibilities and current limitations.
