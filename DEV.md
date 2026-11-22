# VSSS provider development

`overhaul` contains the completed provider/RPC cleanup and P0/P1 functionality
documented in [PARITY.md](PARITY.md). Student facades do not reintroduce the
former provider/worker transport stack.

| Module | Responsibility |
| --- | --- |
| `models.py` | Registered domain dataclasses; one recursive primitive/record codec. |
| `components.py` | GPS freshness, ordinary car methods, per-car locks, ownership and domain events. |
| `_ble_protocol.py` | Existing firmware UUID/layout/validation definitions and typed RobotStatus; no clients or transport runtime. |
| `events.py` / `rpc.py` | Bounded subscriptions, whitelist dispatch, sessions and generic forked connections. |
| `provider.py` | Compose GPS/cars, host/connect lifecycle and disconnect hook. |
| `live.py` | UDP timing validation, HTTP dimensions, asyncio BLE owner, configuration readback and position leases. |
| `scenarios.py` | Validated physical parameters, printable-board default field, typed/two-frame scenario files, seeded goalie generation and export. |
| `physics.py` | Pymunk geometry, finite motor force, identified normal impact, rolling/drift and seeded wobble. |
| `simulation.py` | Delayed commands/observations, existing onboard controller, settings/status, atomic edits and grab ownership. |
| `car.py` / `gps.py` / `runtime.py` | Small student views and cleanup over the provider API; lazy GPS. |
| `input.py` / `teleop.py` | Existing input API, 50 Hz sampling and per-car latest-value workers. |
| `plot.py` | Local setup/artwork, persistent feed and worker-side gestures; EmbeddedPlot exposes frames/pointer input to other GUIs without opening a native window. |
| `recording.py` / `playback.py` | Domain snapshots/events, typed command/status replay and delivery/source time separation. |

## Invariants

- Marked component methods execute normally in-process. RPC resolves only
  provider/gps/cars and invokes the whitelist. A new marked method needs no
  separately implemented client method. Codec decoding never imports arbitrary
  classes; both sides register the shared records on package import.
- Car locks are per ID. Metadata locks never cover BLE/device I/O. Simulation
  edits acquire car locks in sorted order, then the world lock; stepping holds
  the world lock and does not wait for car locks. STOP clears ownership. Owner
  disconnect stops relevant cars concurrently; unrelated owners are rechecked.
- Each teleop worker uses one forked connection to the exact original URI/token,
  or the local object in direct mode. Unsent input is replaced per car. A fast
  car stops without waiting for another worker's radio. Commands are never
  retried automatically against a replacement provider.
- Live connects with an initial zero. Manual writes renew leases; idle
  heartbeats recheck last-write time after taking the per-car GATT lock. Position
  mode sends lease messages, never stale velocity. Failed motion writes drop the
  active command/link instead of resuming previous motion.
- Admin setters stop before mutation. Validate complete configuration and
  canonical readback; full configuration uses bounded contiguous chunks.
  Argument/result payloads are not published, since they may contain secrets.
  Only safe operation names, stopped commands and observed statuses are events.
- Current GPS state is validated using the server's monotonic receipt age plus
  source age; caller limits are checked in the same locked snapshot. Historical
  snapshots remain available explicitly. Replay preserves source capture data
  but refreshes delivery timing. No last-known pose is advertised as current
  after the freshness window closes.
- Simulation advances fixed physical substeps. BLE delay buffers latest pending
  motion; STOP clears delayed work immediately. Camera delay feeds both public
  observations and onboard control. Connection, reading and admin setup do not
  release a waiting scenario. A valid explicit velocity/waypoint does.
- Grabs have session, source and token identity. Held bodies are kinematic and
  cannot receive motion commands; release zeros velocity rather than throwing.
  Scenario replacement invalidates grabs and pending commands. The plot worker
  coalesces adjacent moves while preserving grab/release boundaries. Only
  observation connections reconnect automatically.
- Subscriber queues are bounded. Overflow/failure is inspectable. Recording is
  not yet a transactional snapshot or a server-sequence drain on close (P2).
  Recorded status represents observed historical onboard state, not a new
  hardware freshness assertion during playback.
- Foreground loops use bounded waits so Windows can deliver pending SIGINT.
  RPC disconnect cleanup precedes component teardown; GPS closes even if car
  teardown raises. `@run` preserves the original program error during cleanup.

## Verification

```powershell
python -m unittest discover -s .tests -v
python -m unittest discover -s python_blockly -p test_live_runtime.py -v
python -m unittest discover -s python_blockly -p test_environment.py -v
python python_blockly/verify_export.py boilerplate_velocity.py
python .tests/latency_probe.py --samples 300
```

The feature suite has 71 passing tests and Blockly has 24 runtime tests plus
4 environment/embedded-plot integration tests. The two browser checks are
`node python_blockly/smoke_test.cjs` and `node python_blockly/smoke_environment.cjs`.
Tests use temporary
endpoints, synthetic UDP, fake GATT firmware and separate simulator/student
processes. Student subprocess tests explicitly set VSSS_ENDPOINT so they cannot
accidentally target the lab provider. They do not move hardware. Package tests
verify shared mp controller/config modules and local setup PNG/JSON assets are
present outside the editable checkout.

Simulation capture timestamps advance from a source origin using simulation
time, so fast deterministic tests and source-time controllers see the same
physical intervals. Receipt timing remains wall/monotonic delivery time.
Generated goalie shots use the former rolling coefficient to reach the lane.
The ported example adds small lane/heading corrections for the new dynamic
contact solver; that controller change is separate from physical parameters.

Earlier local measurements were roughly 0.5–0.6 ms median command RPC without
hardware, versus 150 ms median/283 ms max STOP completion including BLE. These
are call-completion measurements, not motor latency. A controlled 80 ms fake
GATT comparison removed 7 redundant heartbeat writes in 15 commands. The fleet
test now additionally holds one radio blocked while another commands/stops.

Hardware smoke tests are explicit (`python .tests/hardware_smoke.py --car 1`)
and use discovery/status/STOP only. Real configuration persistence, navigation,
feedback recovery and multi-car timing use the manual lab acceptance checklist
in PARITY.md. Fitted physics coefficients are available
in examples/physics_lab.json; a different contact solver still needs validation.

The source definitions/marker files in the vision submodule are read locally;
no camera opens just by importing them. Board paths resolve from that submodule
and sprite paths from the marker JSON. Package data includes these assets.
Previous code/ignored experiment data remain recoverable through Git and
`../ipc_sandbox/vsss-legacy-before-cleanup-dc7351f/`.
