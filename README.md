# VSSS IPC experiment

This is the `ipc_cleanup` branch: a deliberately incompatible, minimal rewrite.
One provider composes a `GPS` and a `CarCmd`. The same client reads positions,
sends commands, and assigns callbacks against live or simulated providers.
Playback supplies the same observations and rejects commands.

Python 3.10+ is required. From this checkout:

```powershell
python -m pip install -e .
python -m vsss.simulation examples/scenario.jsonl
```

Installing editable changes the environment's `vsss-*` entry points to this
checkout. To keep the old installation selected, use an isolated virtualenv;
the module commands below also run directly from this directory if Pyro5 and
Bleak are already installed. No camera/OpenCV, gamepad or firmware dependencies
are imported by simulation or playback.

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

The simulator implements bounded kinematic car motion at 30 Hz by default,
with body-frame normalized velocity and no identified motor model. Balls are
stationary until edited. Collision physics, pushing, waypoints, noise, camera
latency and firmware emulation are intentionally outside this first core.

`load_scenario(WorldState)` replaces the complete world and clears commands.
`update_state(StatePatch)` changes only listed car/ball poses; it validates the
whole edit before applying it and stops edited cars. It also accepts a JSON
string containing the tagged patch. Scenario files contain a tagged WorldState
on their first line, or a new-format recording header. A scenario is an initial
state, not a mid-run restoration of timers/random generators.

## Record and replay

Installed commands include `vsss-live`, `vsss-sim`, `vsss-playback` (also
`vsss-bag`), `vsss-observer`, and `vsss-record`:

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
```

The normal tests use synthetic UDP, fake BLE and real subprocess RPC; they do
not move hardware. The explicit hardware check scans, reads status and sends
only STOP. It is not part of automatic test discovery.

The original package and docs are retained locally in `.reference/` (ignored by
Git), and in the pre-rewrite commit. Older tools/examples/tests remain as
reference but are not installed entry points or compatibility guarantees.
See `DEV.md` for module responsibilities and current limitations.
