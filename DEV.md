# IPC cleanup experiment

This branch implements the minimal provider architecture from scratch. The
main checkout is on `ipc_cleanup`; `overhaul` has not been advanced.
There is no commitment to compatibility with the former student
API, CLI tools or schema-3 recording files.

| Module | Responsibility |
| --- | --- |
| `models.py` | Domain dataclasses and one recursive, registered-record codec. |
| `components.py` | Abstract GPS/CarCmd, state publication, ownership and command observation. |
| `events.py` | Ordered bounded per-subscriber queues, explicit overflow. |
| `rpc.py` | One authenticated loopback Pyro endpoint, generic method forwarding, sessions and callback polling. |
| `provider.py` | Compose GPS/cars, describe provider capabilities, host/connect lifecycle. |
| `live.py` | UDP/HTTP vision adapter, one asyncio thread owning lazy BLE connections. |
| `simulation.py` | Physics lifecycle, scenario replacement, atomic position patches. |
| `physics.py` | Pymunk rigid bodies, 240 Hz steps, motors, contacts and rolling drag. |
| `input.py` / `teleop.py` | Previous keyboard/gamepad input API and 50 Hz RPC teleoperation. |
| `plot.py` | Existing BoardPlotter2D artwork, headless output, provider-following viewer. |
| `recording.py` | Protocol-agnostic JSONL snapshots/events and completion markers. |
| `playback.py` | One playback clock dispatching observations into both components. |
| `cli.py` | Foreground provider, recorder and observer entry points. |

Each component has normal Python methods. `@rpc_method` marks the whitelist;
it does not replace local behavior. The client exposes generic component views,
so a new decorated component method needs no corresponding client method.
The RPC endpoint resolves only provider/gps/cars paths and never invokes an
unmarked method. Third-party backend errors are converted to simple error
envelopes; clients receive builtin errors or `RemoteError` with type/message.

Events use server queues and client-side polling on a separate connection. This
keeps arbitrary callbacks in the subscribing process without requiring a
reverse-listening socket. A queue exists before playback's start gate opens.
Polling errors and callback exceptions end the subscription visibly through
`Subscription.error`. No delivery/deduplication promise spans a disconnect.

A persistent root connection identifies a client session. Closing it releases
that client's event queues and stops only cars it currently owns. The generic
client serializes access to its Pyro proxy and explicitly transfers proxy thread
ownership so callback threads can call component methods. Providers own all
component lifetimes; there are no per-car or per-component RPC servers.

## Scope boundaries

- Domain velocity means normalized body-frame x/y and angular w in [-1, 1].
  Pose angles are degrees, counterclockwise from +x; positions are metres in a
  centred field. The simulator maps full-scale commands to 0.6 m/s, 180 deg/s.
- Physics includes contacts, finite motor force, ball pushing, wall rebounds and
  rolling resistance. Constants are illustrative; firmware navigation is absent.
  Manual edits recreate the affected body to discard cached contact impulses.
- Live metadata currently retains dimensions, not the camera/ArUco configuration.
  A fresh timestamp and explicit board-detected flag do not by themselves
  guarantee safe autonomous navigation; consumers must check telemetry age.
- BLE reconnect is explicit/lazy on a subsequent operation, never an automatic
  resumption of motion. Firmware's lease expiry is still the last resort when
  the provider crashes or cannot deliver STOP.
- The first subscriber starts playback. A later subscriber sees current state
  through `gps.state()` and future events, not a replay of missed events.
- Recording starts with a snapshot plus a concurrently established event
  subscription; there may be overlap at the snapshot boundary. It is not a
  transactional snapshot of all components. Events preserve their provider
  sequence; playback uses the recording's relative delivery timeline.
- Shared memory, image arrays, calibration, auto-start, request deduplication,
  plugin discovery and old API adapters are deferred.

Plot uses a read-only background feed with one latest state. It rereads discovery
and reconnects when the endpoint changes or a subscription fails. This retry is
limited to observation: command clients never automatically resend an uncertain
request. Plot RPC timeouts are one second; ordinary client timeouts remain 30s
to accommodate bounded BLE operations. GUI operations stay on the main thread.

Provider foreground loops use short timed waits so Windows can deliver pending
SIGINT. Teardown closes RPC sessions while command components still exist, then
closes components. GPS cleanup runs even if command cleanup fails.

## Development checks

Run `python -m unittest discover -s tests -v` from this worktree. The tests cover
codec roundtrips, validation, atomic edits, queue overflow, authenticated RPC,
callback-to-command calls, unexposed-method rejection, subprocess crash cleanup,
record/playback gating, synthetic UDP decoding and fake-BLE heartbeat/stop.

`tests/hardware_smoke.py --car 1` is a separate, deliberate discovery/status/STOP
check. No nonzero velocity is sent by this check. Keep raw logs in ignored
`.experiment-runs/` and state clearly whether actual hardware checks passed.

## Experiment verification, 2026-11-21

- The original core had 12 passing tests; the expanded suite now has 24,
  including all three provider main loops under pending SIGINT, provider changes,
  original renderer/headless CLI output, 50 Hz teleop and contact physics.
- A wheel built successfully with the new package and console entry points.
- Real car1 advertised over BLE. Outside the sandbox, the new RPC/live adapter
  read status and received a STOP acknowledgement. Mode was idle, Wi-Fi was
  connected, fault was zero and onboard `pose_fresh` was false. No nonzero
  motion command was sent. Bluetooth connection inside the sandbox failed with
  `BleakDeviceNotFoundError`; discovery alone worked there.
- Vision reception and field HTTP polling were tested against synthetic UDP
  and a local HTTP server, not the lab camera server. Real navigation and
  live telemetry freshness have not been validated.

The working CLI baseline is commit `dc7351f`. Cleanup removes obsolete source
from this branch only. Local recordings and firmware backups are recoverable
from `../ipc_sandbox/vsss-legacy-before-cleanup-dc7351f/`; tracked source also
remains available through Git. `mp/`, `aruco_gps/`, the environment, editor
settings and `.hardware_backup/` are retained. No push or `overhaul` merge is
part of this implementation.
