# Overhaul parity status

Updated 2026-11-21 after completing the `ipc_cleanup` experiment. The original
gap review is preserved in commit `07dcc5b`. P0/P1 parity is implemented and the
new provider architecture has been promoted to `overhaul`.

## P0/P1 implementation

| Priority | Capability | Implementation and evidence |
| --- | --- | --- |
| P0 | Waypoints and observable control state | `CarCmd.go_to(ID, Pose)` uses the generic dispatcher. Live writes firmware radians and renews the lease without velocity writes. Sim runs the existing onboard controller with arrival hold, recovery and timeout. `RobotStatus` exposes mode/fault/freshness; velocity, STOP, edits and disconnect cancel navigation. `.tests/test_p0.py`, `test_p1_settings.py`. |
| P0 | Usable feedback | Live ingress validates camera age, source ordering and excess delivery delay. `gps.state()`/`get_car()` omit stale/lost objects; `snapshot()` is explicitly historical. Caller age limits are checked atomically on the provider. Playback preserves capture time and establishes new delivery time. `.tests/test_p0.py`. |
| P1 | Trim/controller/configuration/recovery | Shared validated methods cover reads, mutation/readback, bounded configuration chunks and stopped restart/reconnect. Sim implements matching settings. Admin payloads/results, including credentials, never enter events/recordings. Fake BLE tests exercise actual payloads/readback and lifecycle behavior. `.tests/test_p1_settings.py`. |
| P1 | Moving seeded scenarios and gate | `SimScenario`/`BodyMotion`, two-frame motion derivation, seed/goalie generator, exclusive-create export, hot reload and wait-for-control. Connection/read/admin setup does not release physics. `.tests/test_p1_scenarios.py`, `test_p1_acceptance.py`. |
| P1 | Configurable/identified physics and delay | Convex bodies, motor response, mass, wall restitution, normal impact gain, rolling/drift/roughness parameters, deterministic wobble and BLE/UDP delay queues. `examples/physics_lab.json` carries the former fitted coefficients. Normal impact-gain semantics and analytical rolling distance have regression checks. This is still a new contact solver, not proof of numerical/hardware equivalence. `.tests/test_physics.py`, `test_p1_scenarios.py`, `test_p1_acceptance.py`. |
| P1 | Interactive sim plot | Centre/ball translation, edge rotation, coalesced worker-side requests, exclusive session/source-bound grab tokens, pinned bodies, clamped poses, release on EOF and invalidation on scenario reset. Existing local setup/artwork selection remains. `.tests/test_p1_editing.py`, `test_plot_setups.py`. |
| P1 | Fleet responsiveness | Car locks cover one car instead of holding a fleet lock across I/O. Teleop has independent per-car workers and generic forked connections. The blocked-radio regression verifies a second car receives a command and STOP before the first radio is released. `.tests/test_p1_fleet.py`, `test_latency.py`, `test_teleop_reconnect.py`. |
| P1 | Student/autonomous/Blockly workflows | `Car`, `ClientGPS`, lazy `gps` and `@run` delegate to the new provider; they do not restore the former transport stack. Velocity, observe, waypoint and goalie examples are restored. Blockly discovers the facade signatures and excludes admin/injected-provider options. Actual subprocess velocity/waypoint/goalie tests pass, along with the existing Blockly runtime suite. `.tests/test_p1_students.py`, `blockly_python_demo/test_live_runtime.py`. |

Recordings now also preserve observed typed onboard statuses, so waypoint/status
observations can be replayed. Missing recorded status is reported as unknown.
This is domain data, not the old raw BLE/UDP trace format.

## Verification performed

- 69 core/feature tests pass: codec/RPC/ownership, synthetic UDP, fake BLE
  configuration and leases, shutdown, physics/scenarios/editing, fleet blocking,
  status recording/replay, and student scripts using separate simulator processes.
  The goalie example includes small lane/heading corrections under dynamic
  contacts and defends five deterministic seeded first shots in simulation.
- 24 Blockly runtime tests pass: approved blocks, example programs, editor lease
  expiry, interruption, connection/cleanup failures and multiple cars/gamepads.
- Exported student code is checked against the real new facade with mocked
  provider I/O; velocity-only code keeps GPS lazy.
- Existing local lab setup/artwork assets remain packaged and source-selected.
  Package installation is checked separately from an editable checkout.

These tests do not move real cars. The earlier BLE smoke checks and latency
investigation used discovery/status/STOP, not physical waypoint/configuration or
navigation trials. The user manually confirmed the provider processes communicate.

## Lab acceptance checklist

1. Restart providers/clients after installing this version. Check lab UDP board,
   car and ball observations, then cover/uncover a marker and interrupt the feed;
   stale/missing poses must disappear and recover correctly.
2. Compare direct and RPC teleop; try two cars, takeover, release/STOP and owner
   disconnect. Check that a slow/failing radio does not postpone the other STOP.
3. Read current trim/controller/configuration first. Perform explicit bounded
   write/readback/recovery tests, preserving the lab's real settings. Verify
   restart reconnects stopped and never restores prior motion.
4. Run a short clear-field waypoint; observe arrival and stopped manual takeover.
   Interrupt onboard telemetry and verify firmware recovery/fault behavior.
5. Compare seeded scenarios/contact/rolling behavior with lab recordings. Fitted
   coefficients and the former successful demos do not automatically validate
   the new Pymunk contact solver or controller timing under real delays.
6. Drag/rotate/grab while other objects move; close/change the plot/provider,
   reload a scenario and confirm all grabs/pending commands are released/reset.
7. Record/replay the trial, including waypoint/status events. Run the Blockly
   position, velocity and bounded waypoint examples against the chosen provider.

## P2 intentionally deferred

- Raw BLE/UDP diagnostic payloads, detailed operation timing, and complete old
  analysis-tool reports. New recordings intentionally use the domain schema.
- Transactional multi-component recording snapshots and a sequence drain barrier
  on recorder close. Current startup can overlap snapshot/event data.
- `vsss-dance`, the `calibrate` measurement CLI, old observer convenience flags
  and automatic provider startup/idle exit. Administrative trim/config APIs are
  implemented, but autonomous calibration measurement is a separate tool.
- Periodic HTTP field metadata refresh and automatic command-client reconnection.
  Plot reconnects observations, never replays uncertain motion requests.

Legacy schema-3 bag compatibility and exact old import/signature compatibility
remain explicitly optional. The selected local vision setup/artwork is manually
kept in sync with the lab, as requested. The simulator's identities come from
the scenario; robot identity mutation is an administrative live-firmware feature.
The `GPS` export remains the abstract component; `ClientGPS`/`gps` are student
views. Application code should use those documented surfaces.

The old provider/worker transport layers remain removed. Historical tracked code
is available from commit `8a58b90`; generated recordings/backups/tools remain
recoverable at `../ipc_sandbox/vsss-legacy-before-cleanup-dc7351f/`.
