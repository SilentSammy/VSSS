# ipc_cleanup versus overhaul

Review date: 2026-11-21. Baseline: `overhaul` at `8a58b90`. The working
cleanup baseline before this setup-selection change is `3b07ac9`.
This is a source and test review, not a claim that the new branch has passed
the former branch's hardware or autonomous-behavior acceptance checks.

The basic provider/client architecture works, but **functional parity is not
yet reached**. The largest gaps are autonomous car control and feedback quality,
followed by simulator fidelity and the tools built on those functions. Keep
`overhaul` unchanged until the agreed acceptance cases below pass.

## What is present

| Area | Current behavior |
| --- | --- |
| IPC | One Pyro host, explicit method whitelist, ordinary local component methods, typed records, generic client views and event subscriptions. |
| Providers | Explicit live/simulation/playback processes with a common GPS and car-command interface. |
| Driving | Velocity/STOP, BLE status, discovery, command ownership, owner-disconnect STOP, independent BLE lease renewal. |
| Teleop | Existing input.py keyboard/gamepad layouts, assignments, boosts, 50 Hz input sampling, latest-value handoff, fresh interactive discovery. |
| Direct comparison | Same teleop loop and live components instantiated in one process without an RPC host/client. |
| Plot | Existing BoardPlotter2D; local lab setup selection, marker colors/text/sprites, headless output, automatic provider reconnection. |
| Simulation | Driven square car bodies, car/ball/wall contacts, rolling drag, position edits, hot scenario replacement. |
| Record/replay | Protocol-agnostic typed state/command events, completion markers, speed scaling, loop and first-subscriber start gate. |

The setup work deliberately uses local `aruco_gps.vision_setup.VISION_SETUPS`,
`DEFAULT_SETUP`, the selected setup's marker JSON and local PNGs. It does not
restore the old HTTP setup/artwork distribution mechanism. Plot geometry stays
fixed to the selected setup; a provider-dimension mismatch is displayed.

## Priority gaps

| Priority | Capability in overhaul | Current gap and consequence | Acceptance case |
| --- | --- | --- | --- |
| P0 | Onboard waypoint commands, mode/arrival/fault observation (`Car.go_to`, BLE waypoint/lease handling). | `CarCmd` currently exposes velocity, STOP, discovery and observed commands; live adds raw status. There is no waypoint interface or simulator navigation. Existing autonomous examples cannot perform their primary task. | The same typed waypoint call works with live and sim; arrival/fault/timeout are observable; manual velocity takes over; STOP/disconnect cancels; stale onboard telemetry enters the intended fault/recovery behavior. |
| P0 | GPS feedback usability (`gps.py`: stale_after, camera frame age, out-of-order filtering, excess delay estimate, missing-marker handling). | `GPSState.age()` measures time since reception; `GPSLive._decode()` drops `frame_age_ms` and accepts old/out-of-order packets. A newly received but delayed camera sample can appear fresh. `state()` returns the last snapshot indefinitely, so consumers must currently interpret validity and age themselves. | Synthetic delayed, reordered, board-lost, marker-lost and silent feeds never produce a pose reported as current by the public API. Replay has distinct source time and delivery time. Then repeat against the lab camera. |
| P1 | Trim, controller configuration, persisted robot configuration, reconnect-telemetry and robot restart/readback. | Firmware implementations remain in `mp/`, but their BLE characteristics have no corresponding live adapter methods. Calibration and recovery tools were archived. | Read/write/readback tests with fake BLE first; explicit hardware checks confirm persisted settings and stopped reconnects. Configuration recording must retain the previous protection against leaking credential payloads. |
| P1 | Seeded scenarios, initial velocities, shot generation/export, wait-for-control gate. | Current scenarios contain a first-line WorldState with poses only. No initial car/ball velocities, seed, goalie shot generator, two-frame motion derivation or start gate. A stationary goalie layout is not a moving shot scenario. | Reproduce a saved moving-ball scenario, replay it deterministically, leave it waiting while clients connect, then release physics with the documented command. |
| P1 | Identified ball model, drift/wobble, configurable geometry/motor dynamics, simulated UDP/BLE delay and onboard feedback queues. | Pymunk supplies contact physics, but masses, drag, motor acceleration, body size and wall coefficients are illustrative constants. There is no model import, wobble, latency injection or delayed onboard controller. Old ball-control success results do not transfer to this model. | Port the old contact/rolling/latency cases as behavior tests; compare stopping distance, impact response and timing using the same recorded initial layouts. Keep model changes separate from controller changes. |
| P1 | Plot mouse editing with move/rotate/grab/release and exclusive editor ownership. | RPC position patches exist, but plot has no mouse handler, held bodies or grab lifetime. Objects cannot be manipulated from the field window as before. | Drag centers/balls, rotate edges, keep held bodies anchored, reject a second grab, release on window close/feed change and verify a stale editor cannot modify the replacement provider. |
| P1 | Multi-car responsive control and interruption testing. | Per-car GATT locks are present, but `CarCmd.set_velocity()` still holds one shared ownership lock while `_apply()` waits for BLE; teleop uses one sender connection. A slow car can delay commands to another. Current single-car IPC measurements do not establish fleet timing. | Measure key-to-submit, server queue and GATT completion separately with 2+ cars, including one slow/failing device. Bound stale pending input and verify all relevant STOP paths. |
| P1 | Autonomous demos and student-facing conveniences. | `Car`, the lazy gps facade, Position/Feedback helpers and `@run` are absent. Ball-to-zone, kicking, goalie and repeat-waypoint tools were archived. Blockly's live examples still import the old API and are currently incompatible. | Port one velocity example, one feedback example, a waypoint task and a ball/goalie task using the new API; then migrate Blockly live_runtime/examples. A small facade over the provider may restore ergonomics without restoring the former transport stack. |
| P2 | Raw BLE/UDP diagnostic recording, ownership transitions and transport timings. | New recordings intentionally contain domain state and acknowledged velocity commands, not raw datagrams or GATT start/finish/error events. They cannot reproduce the old BLE timing/ball-analysis reports. Snapshot/event startup is not transactional, and close does not explicitly drain to a known server sequence. | Add optional domain-neutral operation timing/ownership/fault events as needed. Test startup ordering and an immediate command/STOP followed by recorder close; verify a clean recording includes the intended final event. Keep raw protocol tracing an optional adapter diagnostic. |
| P2 | Complete old CLI convenience set. | `vsss-dance` and `calibrate` are absent. Observer print/record are split into observer/record commands; old duration/topic/print flags are absent. Playback remains at the final frame instead of automatically exiting. | Decide which conveniences are still wanted, then test their replacement workflows. These do not all need literal flag compatibility. |
| P2 | Provider lifecycle convenience and consumer recovery. | There is no lazy provider auto-start/idle exit. Plot reconnects; ordinary clients intentionally require explicit reconnect. HTTP field dimensions are fetched once or on refresh rather than periodically. | Agree on explicit versus automatic startup; document recovery and test providers changing while consumers are open. Local setup artwork selection is intentional, not an HTTP-metadata parity requirement. |

## Deliberate differences, not automatic restoration requirements

- Legacy schema-3 recordings and the exact old Python import/signature surface
  were explicitly allowed to break. Keep the new domain format; add an offline
  converter only if old recordings are needed for a concrete experiment.
- `vsss-record` replacing the recording argument of `vsss-observer` is a valid
  simpler workflow. Likewise, local setup/artwork selection is the requested
  replacement for transmitting complete vision setup metadata.
- `vsss-waypoint` was listed as *planned* in the old README and had no installed
  entry point. The missing existing capability is `Car.go_to` and its helpers;
  a new waypoint CLI is optional.
- Array/shared-memory transport, general plugin discovery, seek/pause controls
  and other new features are not prerequisites merely because they might be
  useful. They are outside this parity comparison.

## Proposed route to a verified cleanup

1. Nail down feedback freshness and a typed waypoint/status/STOP interface.
   Add marked component methods so the generic RPC dispatcher continues to work
   without separate handwritten client methods.
2. Port trim/controller/configuration and recovery behavior onto those components.
   Reuse existing firmware/controller logic where appropriate; do not reintroduce
   the former provider/worker IPC layers to regain these methods.
3. Restore moving/seeded scenarios, control-start gating, configurable physical
   parameters and delay injection. Port relevant old simulator regression cases.
4. Restore plot gestures through `update_state` plus explicit grab/release APIs,
   with session ownership and reset-on-provider-change tests.
5. Port representative autonomous examples and Blockly, and decide which
   convenience commands merit restoration. Run their isolated-process tests.
6. Perform manual lab acceptance: actual UDP positions/ball feedback, two-car
   teleop and takeover, direct/RPC comparison, a bounded waypoint, stop on
   disconnect, record/replay the run, and switch providers while plot stays open.
   Review results before advancing `overhaul`.

Passing the cleanup suite verifies the cases it contains, not the old suite's
entire coverage. Earlier real-hardware checks used discovery/status/STOP only;
the recent timing comparison measures call completion rather than motor latency.
The source review also confirms that the firmware and vision submodule are still
present, while old generated data/tools/tests are recoverable in
`../ipc_sandbox/vsss-legacy-before-cleanup-dc7351f/` and tracked code in Git.

## Evidence map

Read the old versions with `git show overhaul:PATH`; current paths are ordinary
files in this checkout.

| Evidence | Relevant paths |
| --- | --- |
| Former public robot/GPS/runtime API | `vsss/car.py`, `vsss/gps.py`, `vsss/_runtime.py` on overhaul. |
| Former BLE protocol, ownership and diagnostic events | `vsss/_ble_protocol.py`, `vsss/_ble_worker.py`, `vsss/_provider_car.py` on overhaul. |
| Former simulation and scenario support | `vsss/_sim.py`, `vsss/_sim_provider.py`, `vsss/_sim_scenarios.py`, `vsss/sim_defaults.json`, `tools/vsss_sim.py` on overhaul. |
| Former plot gestures and tools | `tools/vsss_plot.py`, `tools/_plot_sim.py`, `tools/calibrate.py`, old README/pyproject on overhaul. |
| Current supported methods and semantics | `vsss/components.py`, `vsss/live.py`, `vsss/simulation.py`, `vsss/physics.py`, `vsss/rpc.py`. |
| Current recording/playback guarantees | `vsss/recording.py`, `vsss/playback.py`, `vsss/events.py`. |
| Shared setup/artwork and retained firmware | `aruco_gps/vision_setup.py`, `aruco_gps/marker_config.json`, `mp/main.py`, `mp/onboard_controller.py`. |
| Remaining old integration dependencies | `blockly_python_demo/live_runtime.py`, `fake_devices.py`, `examples/03_rotate_back_and_forth.py`, `examples/05_alternate_waypoints.py`. |
