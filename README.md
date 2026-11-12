# VSSS robot client and firmware

VSSS is the small robot side of the lab system. TE2004B_AD26 owns camera capture,
marker/ball detection, visualization, and pose publication. This repository owns:

- the student `Car`/`GPS` Python API;
- BLE command transport and the manual keyboard/gamepad controller;
- ESP32-C3 motor control and onboard waypoint control;
- guarded velocity/waypoint observation experiments.

Commands travel over BLE. Pose feedback travels over UDP unicast from the lab
vision computer. The default service uses UDP `5000` for data and `5001` for
`POSE_SUBSCRIBE`/`POSE_UNSUBSCRIBE`.

## Install and connect

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Join the `embedded` lab hotspot and enable Bluetooth. Car 1 advertises as `car1`
and uses marker ID 1.

```python
from car import Car

with Car(1) as car:
    print(car.pose)
    print(car.read_status())
    car.set_velocity(x=.2, y=0, w=0)
    car.stop()
    car.go_to(0, .5, 90)  # Onboard controller; the laptop only sends the target.
```

`set_velocity` uses normalized robot-frame forward, left, and counterclockwise
commands. `go_to` sends field X/Y metres and heading degrees to the firmware.

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
`heading_tolerance_deg`, and `hold_ms`.

## Manual control and experiments

The integrated manual controller remains available:

```powershell
python mecanum_client.py
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

## Telemetry check

```powershell
python telemetry_listener.py --server 192.168.12.1
```

Omit `--server` to listen for the legacy broadcast on UDP 5000.

## Firmware and tests

Production firmware files under `mp/` are copied to the ESP32 root. Copy support
modules first, `robot_config.json` next, and `main.py` last. The native USB-safe
`tools/mpremote_no_reset.py` wrapper avoids asserting DTR/RTS on COM17.

Safe fake-BLE and synthetic-telemetry tests never move a physical robot:

```powershell
python -m unittest discover -s tests -v
```
