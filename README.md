# VSSS

Python control and field observation for the lab's mecanum cars. Enable Bluetooth
and join the lab network (`embedded`) to use the real cars and vision feed.
Implementation and maintenance notes are in [DEV.md](DEV.md).

This project is the software platform for a small fleet of ESP32-based mecanum
robots used in the VSSS lab. Students can write Python programs that drive cars,
observe their positions and the ball field, and combine those capabilities for
autonomous behavior. The cars receive commands over BLE, while the shared vision
system publishes marker and ball telemetry over UDP. The repository also includes
provider, replay, plotting, teleoperation, calibration, and development tools.

## Installation

Install Git and Python 3.10 or newer, then clone into any folder:

```sh
git clone --recurse-submodules https://github.com/SilentSammy/VSSS.git my-vsss
cd my-vsss
```

The checkout name is arbitrary. Run the following commands from its root; no
machine-specific paths are needed. If you move the checkout later, rerun setup
to recreate its virtual environment.

Windows (PowerShell):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
./setup.ps1 -Yes
. ./.venv/Scripts/Activate.ps1
python -m pip install -e ".[plot]"
```

Linux:

```sh
bash setup.sh --yes
source .venv/bin/activate
python -m pip install -e ".[plot]"
```

Activate `.venv` again in each new terminal. The vision server runs separately
on the lab computer; students do not need to install or start it.

## Commands

Run any implemented command with `-h` for its arguments.

| Command | Purpose |
| --- | --- |
| `vsss-live` | Run the real BLE/UDP provider until Ctrl+C. Starts automatically when needed if no provider is running. |
| `vsss-bag FILE` | Serve a recorded bag. Waits for the first feed subscriber before playback (`--immediate` skips waiting); exits when playback ends (`--loop` repeats until Ctrl+C). |
| `vsss-sim` | Planned: serve simulated cars and field telemetry. |
| `vsss-observer [FILE]` | Print UDP/BLE activity, or record it to a new bag file with the cached vision setup header. |
| `vsss-plot` | Plot the active provider's field telemetry using its setup metadata. `--persist` waits and reconnects across feed endings and provider restarts. |
| `vsss-teleop` | Discover cars and assign keyboard/gamepad controls. Sends velocity commands only; explicit arguments allow promptless use. |
| `vsss-waypoint` | Planned: send waypoint commands from a terminal. |
| `calibrate --car ID` | Measure and propose trim; optionally save/apply it, apply a saved calibration, or reset trim. Measurement mode moves the car. |

Only one provider mode can run at a time. Start `vsss-bag` before your consumers
to use playback; it exits at the end, or can be stopped with Ctrl+C before switching modes. An automatically
started live provider exits five seconds after its last client leaves.

Teleop's keyboard aliases are `wasd` (Q/E rotation, C boost) and `ijkl`
(U/O rotation, M boost). Examples:

```sh
vsss-teleop --car 1=wasd --car 2=gamepad:0
vsss-observer session.jsonl
```

By default, `vsss-plot` exits when playback ends or its provider disconnects.
Use `vsss-plot --persist` to leave the window open while switching providers.
This mode waits for an existing provider, including at startup, without starting
live mode automatically. Start `vsss-live` or `vsss-bag FILE` separately; each
connection reloads the field setup. Press Q/Escape or close the window to exit.

## Python API

The public modules under `vsss/` are:

| File | Common use |
| --- | --- |
| [car.py](vsss/car.py) | `Car(ID)` connects to a numbered car over BLE. |
| [gps.py](vsss/gps.py) | `gps` lazily starts shared field observation; `GPS()` accesses the same receiver. |
| [input.py](vsss/input.py) | Read keyboard and gamepad state in custom scripts. |
| [provider.py](vsss/provider.py) | `available_cars()` returns car IDs; `ProviderClient` exposes raw observation events for tools. |
| [__init__.py](vsss/__init__.py) | Convenient imports: `Car`, `GPS`, `gps`, `Pose`, `Position`, and `run`. |

### Car

- `set_velocity(x=0, y=0, w=0)`: normalized forward, left, and counterclockwise
  commands in `[-1, 1]`; `stop()` sends zero.
- `go_to(x, y, angle)`: start onboard waypoint navigation using field coordinates.
  Requires healthy telemetry and the car's own marker to be visible.
- `pose` / `position`: current field pose or XY tuple; `None` when unavailable.
  `wait_for_position()` waits for a valid pose.
- `read_status()`: onboard mode, fault, pose, and telemetry counters.
- `read_trim()` / `set_trim(matrix)`: inspect or replace the correction matrix.
  `read_controller()` / `set_controller(parameters)`: inspect or update controller settings.
- `is_connected`: BLE readiness. `close()` / `Car.close_all()`: release this
  script's connections. Prefer `@run` for automatic cleanup.

Velocity control needs only BLE; GPS starts when position is requested.
Multiple scripts can share a car. The latest explicit motion command takes
control, and velocity cancels a waypoint. The motion owner leaving stops the
car; another client's departure does not interrupt it.

### GPS and input

`gps.cars` maps marker IDs to `Pose(x, y, angle)`; `gps.get_car(ID)` returns
one pose or `None`. `gps.balls` contains `Position(x, y)` objects.
`gps.board_detected` indicates board visibility. Stale field observations
return empty collections. Field distances are metres and headings are degrees,
counterclockwise from +X. GPS observation does not require a BLE connection.

For custom input, use `input.is_pressed(key)`, `rising_edge(key)`,
`falling_edge(key)`, and `get_axis(name)`. Gamepad names include `LX`, `LY`,
`RX`, and `RT`; an index suffix selects a device, e.g. `LX1`.

```python
from time import sleep
from vsss import Car, run

@run
def main():
    car = Car(1)
    car.set_velocity(x=0.2)
    sleep(1)
    car.stop()

main()
```

`@run` releases the script's cars and GPS on return, error, or Ctrl+C.
Starting scripts: [velocity](boilerplate_velocity.py),
[field observation](boilerplate_observe.py), and
[autonomous control](boilerplate_autonomous.py).

## Blockly
