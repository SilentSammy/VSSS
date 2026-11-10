# Overhaul branch handover

This document describes the repository as it exists on the `overhaul` branch after
the BLE, vision, calibration, and onboard-waypoint experiments. It is intended as
the starting point for a cleanup pass. Do not assume every older README statement
is still true; the known documentation conflicts are listed below.

The plan to move the improved vision/telemetry features back into TE2004B_AD26 and
delete VSSS's duplicate is in [docs/te2004b-upgrade/README.md](docs/te2004b-upgrade/README.md).

## Current state

- Branch: `overhaul`
- Last shared commit at handover: `2b066b9` (`did a bunch of control optimizations`)
- Actual robot: car 1, ESP32-C3 on `COM17`, BLE name `car1`, ArUco marker 1.
- Lab hotspot: `embedded`; the ESP32 configuration currently contains its password.
- Vision computer: `/home/aavila/Desktop/VSSS` on the lab desktop, normally reachable
  as `aavila@192.168.12.1` on the hotspot or `aavila@100.87.82.76` through Tailscale.
- Normal feedback service: UDP 45888 data, UDP 45889 subscription/discovery.
- The ESP32 currently listens for the legacy/broadcast copy on UDP 45888. Student
  laptops use the more reliable unicast subscription service.
- COM17 was last flashed with the files under `mp/`, joined WiFi at
  `192.168.12.253`, advertised `car1`, and accepted manual and waypoint BLE commands.
- Physical testing is suspended because the user is away from the lab.

## System in one page

There are two independent transports presented as one student-facing robot API:

```text
Student Python process
  Car(1)
    |-- commands -> MecanumBLEClient -> ble_worker subprocess -> BLE -> ESP32-C3
    `-- feedback -> shared GPS singleton <- UDP unicast <- lab vision desktop

ESP32-C3
  BLE manual velocity or waypoint command
  + UDP broadcast pose for its own marker
  + saved correction matrix/controller parameters
  -> motor mixing/PWM
```

The intended minimal student surface is:

```python
from car import Car
from gps import GPS

gps = GPS()
with Car(1) as car:
    pose = car.pose
    car.set_velocity(x=.2, y=0, w=0)
    car.go_to(0, .5, 90)  # starts the onboard waypoint controller
    car.stop()
```

`Car` hides the fact that BLE commands and UDP feedback are unrelated networks.
`GPS` is one receiver per process and exposes every visible marker and ball.

## Runtime components

### Student/laptop side

- `car.py`: preferred high-level API. Owns a `MecanumBLEClient`, reuses the `GPS`
  singleton, exposes `pose`, `position`, `set_velocity`, `go_to`, `stop`, and trim.
- `gps.py`: UDP receiver, subscription renewal, freshness/packet-age checks, immutable
  `Pose`/`Position` values, all cars and balls.
- `mecanum_client.py`: synchronous BLE client facade plus the manual demo. BLE itself
  runs in a child process. The demo starts in WASD/gamepad mode; `P` enters a terminal
  waypoint mode accepting `x y angle`, commas or spaces, `stop`, `manual`, and `quit`.
- `ble_worker.py`: the sole Bleak owner. It coalesces velocities, sends heartbeats,
  forwards waypoint/configuration writes, reconnects stopped, and exits on parent EOF.
- `user_input.py`: pynput keyboard and `inputs` gamepad state.
- `main.py`: old minimal student boilerplate that rotates briefly. It is not the
  vision server; that entry point is `detection/main.py`.

### ESP32 firmware (`mp/`)

Production files currently required on the board root are:

- `main.py`: boot orchestration, WiFi, BLE callbacks, mode/safety state machine,
  onboard waypoint loop, status characteristic.
- `ble_server.py`: one GATT service, one connection at a time, advertisement,
  heartbeat timeout, disconnect callback.
- `mecanum.py`: motor/PWM implementation and wheel mixing.
- `axis_trim.py`: one 3x3 command-correction matrix.
- `onboard_controller.py`: predictive field-to-robot waypoint controller.
- `pose_receiver.py`: polling UDP JSON receiver for the car's marker.
- `robot_config.py` and `robot_config.json`: validated persistent identity, WiFi,
  trim matrix, and controller parameters.
- Existing `boot.py` can remain unchanged.

Important boot ordering: the firmware associates WiFi before importing
`ble_server`, because activating BLE earlier caused ESP32-C3 coexistence problems.
The tested compact board also uses WiFi `txpower=5`.

### Lab vision (`detection/`)

- `detection/main.py`: board/marker/ball processing and telemetry publication.
- `camera_stream.py` / `frame_stream.py`: camera acquisition and viewer plumbing.
- `board_*`, `marker_tracker.py`, `ball_det.py`: geometry and detection.
- `telemetry.py`: broadcast plus expiring unicast subscribers.
- `plotter2d.py`: field UI, including balls.
- `run.sh`: normal supervisor/launcher; calls `configure_lab_network.sh` first.
- `vision_setup.py`, `marker_config.json`, resources: setups and printed markers.

Useful low-latency launch on the lab desktop:

```bash
cd /home/aavila/Desktop/VSSS
bash detection/run.sh --setup setup_160cm --headless \
  --no-ball-detection --no-camera-viewer --rate 30
```

Omit the headless/no-ball/no-viewer flags when the UI or ball is required. Do not
run TE2004B_AD26 and VSSS against the same camera simultaneously.

## BLE protocol

All characteristics share service UUID
`12345678-1234-5678-1234-56789abcdef0`.

| Suffix | Purpose | Payload |
|---|---|---|
| `def1` | manual velocity | 3 signed int8 bytes: robot x/y/w |
| `def2` | lease heartbeat | 1 byte; response contains current mode |
| `def3` | correction matrix | `<9f`, row-major output rows / input columns |
| `def6` | waypoint | `<fff`: field x metres, y metres, heading degrees |
| `def7` | one-shot control | byte 0 stop/cancel; byte 1 flush configuration |
| `def8` | controller configuration | `<11f` in `CONTROLLER_KEYS` order |
| `def9` | status | `<BBBBfffI`, 20 bytes |

Status fields are mode, flags, fault, reserved, x, y, heading-degrees, packet count.
Modes: idle 0, manual 1, position 2, reached 3, fault 4. Faults: none 0,
telemetry 1, timeout 2. The host currently does not expose this status cleanly to
students; doing so would make waypoint failures much easier to diagnose.

Safety behavior:

- A zero velocity explicitly releases the motor PWM outputs.
- BLE disconnect immediately stops the motors.
- Missing BLE writes for 1.5 s stops and disconnects; initial GATT setup gets 10 s.
- Position mode stops on stale/missing pose, WiFi loss, or 30 s waypoint timeout.
- Manual velocity remains active while the worker is alive and renewing it.
- A new waypoint replaces the current waypoint without needing a manual-mode reset.

## Onboard waypoint controller

The ESP32 receives its own marker pose by UDP and transforms field-frame position
error into robot-frame x/y/w. Saved car-1 values currently include `kp_dist=4`,
`kp_heading=.55`, maximum translation `.4`, maximum yaw `.35`, prediction horizon
`.2 s`, and a 2.5 cm / 5 degree arrival band held for 600 ms.

Two bugs were fixed immediately before handover:

1. The controller used to recompute at the 100 Hz firmware loop rate even though
   camera poses arrive around 10-15 Hz. That made a new frame's displacement look
   as though it occurred in 10 ms and produced huge derivative estimates. It now
   updates only once per new `frame_sequence`, using the pose receipt timestamp,
   and holds the motor command between frames.
2. The hard minimum translation/yaw command is no longer applied directly outside
   the arrival band; it only applies beyond twice the tolerance, reducing the
   near-target limit cycle.

The controller successfully moved car 1 under a foreground REPL diagnostic from
approximately `(0.188, 0.741, 92 deg)` to `(0.016, 0.020, 90 deg)` for a
`(0, 0, 90 deg)` target. It then safely stopped because marker telemetry became
stale just before completing the arrival hold.

### Pinned issue: intermittent drunken/zigzag waypoint runs

Before leaving the lab, the user observed that identical waypoint moves sometimes
looked excellent and sometimes zigzagged as if control updates were dropping.
The frame-rate mismatch fix above was flashed, but there was not enough physical
testing afterward to declare this resolved.

Do not start by blindly retuning P gains. First record, on the same timebase:

- every received frame sequence, receipt interval, server `frame_age_ms`, and pose;
- every controller update and x/y/w output;
- mode/fault changes and BLE lease timing;
- missing/duplicate/out-of-order frame sequences and marker visibility;
- WiFi RSSI if cheaply available.

Likely remaining contributors are intermittent marker detection, UDP broadcast loss
on the ESP32 while BLE is active, variable camera processing gaps, the old `.2 s`
prediction horizon being too large for onboard control, and drivetrain dead-zone
compensation. The ESP32 still uses broadcast telemetry while laptops use unicast;
moving the ESP32 to an explicit unicast subscription is a promising experiment.
Use recorded frame timing before selecting a new horizon.

## Calibration

- `calibration/cars.json`: accepted per-car 3x3 correction matrices.
- `calibration/controller_car1.json`: laptop-controller profile from experiments.
- `mp/robot_config.json`: firmware's deployable car-1 matrix/controller/WiFi config.
- `tests/experiments/`: reusable physical experiment package, safety bounds,
  recording, fitting, assessment, and controller comparisons.
- `calibrate_controller.py`: laptop-side parameter fitting/run entry point.

The correction matrix is now persisted on the robot by `RobotConfig`. The host
`Car` class also writes `calibration/cars.json` after connecting, which is redundant
and can overwrite the robot's persisted matrix. Cleanup should choose the robot as
the single source of truth and make host-side calibration an explicit provisioning
operation rather than an automatic connection side effect.

## Confusing duplication to resolve

There are currently two waypoint implementations:

1. Preferred new path: `Car.go_to()` -> BLE waypoint -> `mp/onboard_controller.py`.
2. Older path: root `go_to.py` + `position_controller.py`, running the predictive
   loop on the laptop and repeatedly sending manual velocities.

The manual demo's waypoint console uses the new onboard path. Root `go_to.py` still
uses the old laptop path. Decide which behavior is retained; do not present both as
the canonical student interface. The project's stated direction is onboard control,
with robot-specific calibration stored on the robot.

`mp/hybrid_probe.py` is a WiFi/BLE coexistence prototype, not production firmware.
`tests/experiments/hybrid_ble_probe.py` belongs to that investigation.

## Cleanup proposal

Cleanup now has two repositories and must be staged. The complete upstream design,
process topology, IPC/network boundary, migration order, and deletion acceptance
criteria are in [docs/te2004b-upgrade/README.md](docs/te2004b-upgrade/README.md).

### Target ownership after consolidation

Keep in VSSS as the minimal robot product:

- `car.py`, `gps.py`, `mecanum_client.py`, `ble_worker.py`, `user_input.py`
- `mp/main.py`, `ble_server.py`, `mecanum.py`, `axis_trim.py`,
  `onboard_controller.py`, `pose_receiver.py`, `robot_config.py`
- core fake-BLE, student-API, telemetry-client, firmware-config, and controller tests
- one concise student example and one operator/deployment guide

Move to or retain in TE2004B_AD26:

- camera and latest-frame service;
- board/marker pose estimation;
- optional ball detector and ball projection;
- local camera annotations, field plot, and browser viewer;
- broadcast plus expiring unicast telemetry publisher;
- vision resources, camera/board configurations, tests, and benchmarks.

VSSS should consume TE2004B only through the documented UDP protocol. It should
not import TE2004B source or keep a vendored copy once migration is complete.

### Keep temporarily during migration

- The entire VSSS `detection/` tree and its tests/resources. It is the working
  reference implementation until TE2004B reaches feature parity.
- Compatibility tests that compare TE2004B packets/poses with the VSSS reference on
  identical recorded or synthetic frames.
- Both legacy broadcast and new unicast client paths until the ESP32 subscription
  experiment is validated on the lab hotspot.

Do **not** simplify by deleting `detection/` first. Upstream and validate these
features in TE2004B before removing the copy:

1. Expiring unicast subscriptions while preserving legacy broadcast.
2. Deadline-based telemetry scheduling that does not accidentally halve the rate.
3. Optional ball detection, projection, packet field, and plotting.
4. Headless, no-ball, and no-viewer performance modes.
5. `schema_version`, `stream_id`, frame sequence, and frame-age semantics.
6. Optional process isolation: camera, latency-critical pose, best-effort ball, and
   visualization workers supervised by one launcher.

Visualization corners/contours/images are local IPC only. Public UDP remains compact
field telemetry. If process isolation is adopted, use a bounded sequence-keyed raw
frame ring and local geometry results; never enlarge WiFi packets with drawing data.

### Keep but move behind an explicit development/experiments boundary

- `tests/experiments/`, `calibrate_controller.py`, calibration analysis profiles
- hardware diagnostics and repeatable track tests
- simulator file `vss_scene.ttt` if it still has an owner/use case

### Candidates to remove or archive after review

- `mp/hybrid_probe.py` and hybrid investigation code
- root laptop waypoint controller once onboard control is accepted
- VSSS `detection/` and duplicate vision resources, but only after every acceptance
  criterion in the TE2004B upgrade README passes
- generated/extra marker PDFs and sprites not needed by canonical TE2004B setup
- one-off files under `tools/` after extracting any generally useful diagnostics
- stale top-level `todo.txt`

Do not delete experimental recordings or user assets merely because they are large;
review Git history and confirm what is reproducible first.

### Cleanup gates

Before deleting the copied vision stack, confirm all of the following:

- TE2004B marker coordinates/headings match the VSSS reference on the same frames.
- Balls can be enabled or disabled without changing consumer compatibility.
- Multiple unicast clients coexist with legacy port-5000 broadcast clients.
- The ESP32 receives reliable subscribed telemetry while BLE is active.
- Pose publication is never blocked by slow ball or visualization work.
- Local annotation geometry is matched by stream/frame ID and never broadcast.
- Headless/no-viewer mode performs no GUI/JPEG work.
- TE2004B vision tests and VSSS protocol-client tests pass.
- VSSS production imports no module from its old `detection/` directory.

Delete duplicated files in a dedicated, reviewable commit after these gates pass,
not in the same commit that introduces the replacement service.

## One-off diagnostic tools currently present

- `tools/mpremote_no_reset.py`: necessary workaround for this ESP32-C3's native USB
  Serial/JTAG port. Stock mpremote asserts DTR/RTS on open and can reset COM17 into
  the ROM loader. This wrapper opens with both inactive.
- `tools/check_onboard_telemetry.py`: read-only ESP32 WiFi/UDP pose check.
- `tools/repl_waypoint_test.py`: foreground MicroPython movement test with a fixed
  target and `finally: car.stop()`; it interrupts production `main.py` while running.
- `tools/check_waypoint_ble.py`: direct Bleak current-pose waypoint/status probe.
- `tools/ble_waypoint_test.py`: host BLE waypoint plus UDP observation.

The last two could not connect from the agent's noninteractive Windows execution
session even while the user's interactive PowerShell `mecanum_client.py` worked.
This was a WinRT execution-session limitation, not a demonstrated robot fault.

## Flashing COM17

Back up the board first. Copy the complete production file set listed above, with
`mp/main.py` copied last, then reset. A local pre-overhaul board backup was placed
under ignored `.hardware_backup/` during development.

Example using the native-USB-safe wrapper:

```powershell
python tools/mpremote_no_reset.py connect COM17 fs cp mp\ble_server.py :ble_server.py
python tools/mpremote_no_reset.py connect COM17 fs cp mp\mecanum.py :mecanum.py
# ...copy every production support module and robot_config.json...
python tools/mpremote_no_reset.py connect COM17 fs cp mp\main.py :main.py
```

Expected boot output includes:

```text
BLE Server started with 7 controls
Advertising as car1
car1 ready; WiFi 192.168.12.x; control onboard
```

Opening/closing native USB can leave DTR/RTS in surprising states. If the ROM says
`DOWNLOAD(USB/UART0/1)`, release BOOT and press RESET/EN, then use the wrapper. Do
not issue movement during a flash verification.

## Tests

Safe full suite (fake BLE/synthetic vision, no physical movement):

```powershell
python -m unittest discover -s tests -v
```

The final focused run after the frame-timing/retarget fixes was:

```powershell
python -m unittest tests.test_onboard_controller tests.test_ble_upgrade tests.test_robot_config
```

It passed 30 tests. The full suite was then run at handover and passed **99 tests
with 2 skipped** in about 42 seconds. Two experiment tests emitted existing
`ResourceWarning` messages for unclosed CSV readers, but the suite exited cleanly.

Physical experiment commands are preview-only unless `--execute` is supplied. Read
`tests/experiments/README.md` before using them.

## Documentation known to be stale

Update or replace `README.md` and `BLE.md` during cleanup. Known incorrect passages:

- `BLE.md` lists only three firmware files; production now needs seven support files
  plus configuration.
- `BLE.md` describes the waypoint-less protocol and says trim is RAM-only.
- `README.md` also says trim resets on reboot; firmware now persists it.
- `README.md` says the ESP32 needs no WiFi; onboard waypoint control requires WiFi.
- `README.md` presents root `go_to.py` as canonical even though it is the older
  laptop-side controller.
- Some comments still describe earlier names (`Eq`, teams) or pre-overhaul behavior.

Prefer one short architecture document, one student quick-start, and one lab operator
guide over continuing to append historical behavior to the root README.

## Recommended next-agent sequence

1. Do not move hardware until the user returns to the lab.
2. Read both this handover and the TE2004B upgrade README before moving files.
3. Run both repositories' test suites and inventory imports/references before any
   deletion or module relocation.
4. Create a TE2004B feature branch and upstream unicast plus the deadline scheduler
   first, preserving port 5000 broadcast compatibility.
5. Benchmark the combined vision loop, then isolate visualization. Add a ball worker
   only if measured p95/p99 latency, frequency, or frame-gap improvements justify it.
6. Add optional balls, compact versioned telemetry, timing/stream identity, and the
   local sequence-matched visualization path.
7. Switch one ESP32 from broadcast to subscription mode and record sequence gaps,
   update jitter, stale faults, and waypoint paths before/after.
8. Reproduce the good and zigzag runs with frame-gap/controller logging before
   retuning the prediction horizon or gains.
9. Establish onboard waypoint control and robot-persisted calibration as the
   canonical production path; retain laptop control only as an experiment tool.
10. Make robot configuration authoritative and remove automatic host trim writes.
11. Add host-readable status/mode/fault reporting.
12. After the cleanup gates pass, delete VSSS `detection/` in a separate commit and
    rewrite the student quick-start and lab operator documentation around TE2004B.
13. Finally archive/remove obsolete experiments and one-off tools with explicit
    review, leaving a small product tree and a clearly separated development tree.
