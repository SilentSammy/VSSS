# VSSS robot client and firmware

VSSS is the small robot side of the lab system. The shared `aruco_gps`
submodule provides camera capture, marker/ball detection, visualization, and
pose publication. This repository owns:

- the student `Car`/`GPS` Python API;
- BLE command transport and the manual keyboard/gamepad controller;
- ESP32-C3 motor control and onboard waypoint control;
- guarded velocity/waypoint observation experiments.

Commands travel over BLE. Pose feedback travels over UDP unicast from the lab
vision computer. The default service uses UDP `5000` for data and `5001` for
`POSE_SUBSCRIBE`/`POSE_UNSUBSCRIBE`.

## Install and connect

Clone the repository and run `./setup.ps1` on Windows or `bash setup.sh` on
Linux. Add `-Yes` or `--yes` to skip prompts, and `-Vision` or `--vision` to
also install the optional shared vision service. The scripts initialize the
submodule. Users only consuming the lab feed do not need to run that service.

To start a local service on Linux, use `bash aruco_gps/run.sh`. The lab desktop
normally runs one standalone instance shared by VSSS and TE2004; see
[`aruco_gps/README.md`](aruco_gps/README.md). Network setup is separate.

For manual client installation:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Join the `embedded` lab hotspot and enable Bluetooth. Car 1 advertises as `car1`
and uses marker ID 1.

The student API is in the `vsss` package: import `Car`, `gps`, and `run` from `vsss`.
Keyboard and gamepad helpers are available as `from vsss import input`.
Legacy `from car import Car` and `from gps import GPS` imports remain available
after installation for existing scripts.

Root-level starting scripts: [velocity only](boilerplate_velocity.py),
[field observation](boilerplate_observe.py), and
[autonomous control](boilerplate_autonomous.py). The velocity and autonomous
examples move car 1 when run; edit their commands/target for your exercise.

```python
from time import sleep
from vsss import Car, run

@run
def main():
    car = Car(1)
    car.set_velocity(x=.2)
    sleep(1)

main()
```

`@run` stops/disconnects all cars created in this process and releases its GPS
connection when the function returns, raises an error, or receives Ctrl+C. Call
the decorated function normally; it is a synchronous script entry point. Multiple
cars can be created in the same function. Cleanup does not affect other scripts.

Car construction and velocity commands use BLE only. Position access (`car.pose`,
`car.position`, or `car.gps`) starts the shared GPS receiver lazily. Missing camera
telemetry does not prevent velocity control; a missing/stale pose returns `None`.

For field-only observation, no car or BLE connection is needed:

```python
from time import sleep
from vsss import gps, run

@run
def main():
    while True:
        print(gps.cars)   # {id: Pose(x, y, angle)}
        print(gps.balls)  # tuple of Position(x, y)
        sleep(.1)

main()
```

Importing `gps` does not start reception; its first field access does.
`GPS(...)` remains available for explicit configuration before first access.
`Car.close()`, `with Car(...)`, and `Car.close_all()` in a `finally` block are
also supported. `Car.close_all()` attempts every car even if one cleanup fails.

`set_velocity` uses normalized robot-frame forward, left, and counterclockwise
commands. `go_to` sends field X/Y metres and heading degrees to the firmware.
Waypoint control requires healthy onboard telemetry and visibility of the car's
own marker. `car.wait_for_position()` waits for desktop visibility when your
planning code also needs field feedback; `car.read_status().pose_fresh` checks
the robot's separate onboard reception.

The firmware disables Wi-Fi modem sleep for lower-latency feedback. Its
freshness check includes reported camera frame age and time since the last
unique frame; duplicates do not keep an old pose alive. A short loss stops the
motors, retains the waypoint, and resumes from a reset velocity estimate only
after fresh telemetry returns. Arrival latches translation and heading inside
their tolerances, stops those axes immediately, and only reactivates them after
leaving configurable hysteresis bands. This prevents loaded deadband stalls and
single-wheel chatter while preserving a continuous settled hold before success.
If heading leaves its band during translation, translation pauses while heading
is reacquired in place; the controller never mixes a large yaw correction with
translation near this unstable coupled condition.

## Stored robot configuration

Trim and controller configuration live on the robot. Connecting does not rewrite
them. Configuration can be read, partially updated, and read back over BLE:

```python
with Car(1) as car:
    print(car.read_trim())
    print(car.read_controller())

    car.set_trim(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    car.set_controller({"horizon": .15})

    status = car.read_status()
    print(status.mode_name, status.fault_name, status.pose_fresh)
```

If telemetry needs recovery, `car.reconnect_telemetry()` stops the robot and
restarts its Wi-Fi and UDP receiver. `car.restart()` stops and reboots it, then
waits for BLE to reconnect in the stopped state. Both require a working BLE
connection. Neither resumes a prior velocity or waypoint.

Controller keys are `kp_dist`, `kp_heading`, `max_speed`, `max_w`, `horizon`,
`velocity_alpha`, `min_speed`, `min_w`, `position_tolerance`,
`heading_tolerance_deg`, `hold_ms`, `position_hysteresis`,
`heading_hysteresis`, `telemetry_recovery_ms`, and `waypoint_timeout_ms`.
Hysteresis values are exit-band multipliers (at least 1); the two timing values
are milliseconds. Partial updates remain supported and are stored on the robot.

## Manual control and experiments

The integrated manual controller remains available:

```powershell
vsss-manual
```

`P` switches between keyboard/gamepad control and the onboard waypoint console.

The experiment runner reads robot configuration/status and records pose plus
firmware mode/fault. Motion is preview-only unless `--execute` is present:

```powershell
python -m tools.run --waypoint 0 .5 90
python -m tools.run --waypoint 0 .5 90 --execute
python -m tools.run --velocity .2 0 0 --duration 2 --execute
python -m tools.run --observe 10
python -m tools.run --trim trim.json --controller controller.json
```

Calibration runs signed velocity pulses around the field center `(0, 0, 90)`
by default and saves its frames, original settings, and trim proposal. For the
supervised upper-half track, specify `--center 0 .75 90`:

```powershell
calibrate --car 1
calibrate --car 1 --auto-apply
calibrate --car 1 --ask-apply --save-json proposal.json
calibrate --car 1 --center 0 .75 90 --translation-level .5 --yaw-level .2 --pulse-seconds .8 --yaw-seconds .7
calibrate --car 1 --apply proposal.json
calibrate --car 1 --apply tools/runs/RUN/original.json
calibrate --car 1 --reset
```

Each measurement prints the proposal and saves a recording in `tools/runs/`;
`--save-json` also saves it at a chosen path. `--auto-apply` applies a passing
proposal without prompting, while `--ask-apply` requests confirmation. Apply a
saved proposal or original trim with `--apply`, or reset to the identity matrix
with `--reset`; these actions do not move the car. The script stops on lost
telemetry or boundary risk; controller gains are not retuned by these pulses.
The guard follows the selected center within field margins; the car must start
inside that guarded region. Rotation pulses default to 0.7 seconds and remain
subject to the 70-degree rotation guard, including during settling.

Velocity trials stop if pose telemetry becomes stale or the car crosses configured
bounds. Waypoint trials send one target to the firmware and observe its BLE status;
they do not run a second controller on the laptop.

## Observation and recording

`vsss-observer` shows a readable UDP/BLE feed. Add `--output` to also record
full events, or `--quiet --output` to record without displaying each event.
Both topics are observed by default; `--topics udp` or `--topics ble` selects one.

```powershell
vsss-observer
vsss-observer --topics udp --duration 10
vsss-observer --topics ble
vsss-observer --quiet --output session.jsonl --duration 60
vsss-observer --topics udp --broadcast --port 5000
```

UDP defaults to unicast from `192.168.12.1`; use `--server` for another host
or `--broadcast` for passive broadcast reception. Unicast uses an automatic
local port unless `--listen-port` is supplied. `--subscription-port` overrides
the server subscription port (default: data port + 1).

The shared local VSSS Provider starts automatically when a car client, GPS, or
observation tool needs it. Each car has one controller at a time; if that controller
exits, the provider stops and releases the car even when a recorder is watching.
The provider exits after its last client leaves and a short idle period.

The provider owns the BLE connections and UDP sockets. GPS and tools using the
same vision source settings share one subscription. GPS turns those packets
into student-facing poses and ball positions; recording tools observe original
datagrams and BLE operations through private, independent event streams.

Start the observer before the programs whose BLE traffic you want to capture.

The JSON Lines file combines original UDP datagrams with local provider BLE reads,
writes, acknowledgements and errors. BLE events include a car, control session,
connection and request ID; velocity writes distinguish commands, heartbeats,
and stops. Timestamps use the local monotonic and wall clocks in nanoseconds.
UDP events include the exact bytes in `payload_base64`, sender, provider receive
timestamps, source ID, sequence, and validation result. Invalid packets are
preserved too. All observers of a source receive the same event timestamps and
sequence numbers. A sequence starts with each new source, not each recording.

`session_end.complete` is false if recording ends due to an error; an abrupt
termination can leave no end event. The recorder sees BLE traffic from VSSS
clients on this computer, not other computers or applications that bypass the
provider. It does not replay operations sent before it starts. A slow observer
gets an overflow event or disconnect; it cannot block car control or other
observers. GPS automatically reconnects after provider interruption and expires
old positions; a recorder ends incomplete so interruptions remain visible.
Run `vsss-observer --help` for all options. The observer is an ordinary
provider consumer.

For development analysis, `python -m tools.ball_dynamics BAG.jsonl --car 1`
extracts car/ball contacts, rejects detector outliers and sustained pushes,
fits an impact-plus-rolling model, and reports leave-one-roll-out endpoint
errors. Add `--output model.json` to preserve the fitted parameters and episode
diagnostics. This is an initial free-roll model: wall rebounds and prolonged
dribbling are identified but not simulated.

## Live and recorded field plotting

Observation consumers use a small provider contract: `snapshot`,
`read(timeout)`, `finished`, and `close()`. `vsss.provider.LiveProvider` reads
the real shared VSSS Provider, while `BagProvider` replays a `vsss-observer`
JSONL file with the same event shape and relative timing. A future simulation
provider can implement the same contract without special cases in plotters or
analysis tools.

Install the optional GUI dependencies with `pip install -e ".[plot]"`, then:

```powershell
vsss-plot                         # live unicast telemetry
vsss-plot --broadcast             # live broadcast telemetry
vsss-plot --bag session.jsonl     # real-time bag replay
vsss-plot --bag session.jsonl --speed 4
vsss-plot --bag session.jsonl --speed 100 --output final-frame.png
vsss-plot --duration 10 --output live-snapshot.png
```

The renderer is adapted from TE2004B_AD26's OpenCV board plotter but accepts
the published VSSS marker/ball packets directly, without camera, ArUco, or
board-generation dependencies. Press Q, Escape, or close the window to stop.

## Firmware and tests

Firmware records waypoint diagnostics to `navigation.jsonl` (newest),
`navigation.1.jsonl`, and `navigation.2.jsonl`, limited to 128 KiB each. The
JSONL schema header names the fields in each sample's `v` array and includes
the boot ID, reset cause, controller parameters, and trim. Samples include
onboard pose/timing, target, estimated velocity, raw/trimmed commands, actual
PWM frequencies/duties, loop time, and free heap.

During motion, the logger uses a preallocated RAM ring (up to 10 KiB / 13 seconds
at 5 Hz, reduced when heap is tight) and performs no flash writes. The schema
records its actual capacity, and 32 KiB of heap is reserved for control/networking.
Stop, arrival, fault, disconnect, and clean
shutdown save the trace. Idle time does not replace navigation samples. To
preserve a reproduced problem, stop/disconnect the car while it is still
powered before switching batteries or bringing it back for USB inspection.
Sudden power loss during motion can lose the current RAM trace; earlier saved
waypoints survive reboots. A `flush` event reports any overwritten RAM records.
Storage errors disable logging while leaving control operational.

Production firmware files under `mp/` are copied to the ESP32 root. Copy support
modules first, `robot_config.json` next, and `main.py` last. The native USB-safe
`tools/mpremote_no_reset.py` wrapper avoids asserting DTR/RTS on COM17.

Safe fake-BLE and synthetic-telemetry tests never move a physical robot:

```powershell
python -m unittest discover -s .tests -v
```
