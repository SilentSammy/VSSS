# VSSS

Robot commands use BLE; position feedback comes from the lab camera over UDP.
See [BLE.md](BLE.md) for the robot/client code.

## Repository layout

- `detection/`: lab-side camera service, marker/ball detection, plotting,
  calibration, resources, and launch scripts.
- `car.py`, `gps.py`: unified student interface and shared position receiver.
- `mecanum_client.py`, `ble_worker.py`, `user_input.py`: BLE control and manual input.
- `telemetry_listener.py`: lightweight student-side position receiver/debugger.
- `mp/`: ESP32 robot firmware.
- `tests/`: regression tests for both sides.
- `tests/experiments/`: reusable recording, correction comparisons, strides, and waypoint
  experiments; see [the experiment guide](tests/experiments/README.md).
- `calibration/runs/`: historical experiment recordings, retained at their original paths.

The root `main.py` is a minimal student starting point: it connects to car1,
waits for its position, then rotates briefly and stops. Run `python main.py`
on the lab hotspot. Replace its short control loop with your own logic. The
vision entry point is `detection/main.py`; importing it cannot shadow a student's
`main.py`. Vision dependencies are installed separately from root requirements.

## Student interface

Install root `requirements.txt` into your Python environment. Join the lab
hotspot for camera feedback and enable Bluetooth for control. Windows and Linux
use the same code. Start the interactive waypoint controller once:

```bash
python go_to.py
```

After it connects, enter field X/Y in metres and heading in degrees CCW. The
connection remains open between commands:

```text
waypoint> 0 0.75 90
waypoint> -0.2 0.9 135
waypoint> quit
```

`go_to.py` uses the calibrated predictive controller, stops between waypoints
and on stale telemetry, and always stops before disconnecting.

Controller calibration can be regenerated from a prior pulse recording, or run
end-to-end with guarded physical pulses. `--save` updates the per-car profile
that `go_to.py` loads on its next launch:

```bash
python calibrate_controller.py tests/experiments/runs/RUN/samples.csv
python calibrate_controller.py --run --car 1 --save
```

Put a custom control loop in `main.py`:

```python
import time
from car import Car
from gps import GPS

gps = GPS()  # Shared receiver; also created automatically by Car.
with Car(4) as car:  # Connects to car4; camera marker ID must be 4.
    print(car.trim_matrix)
    car.set_trim(((.95, 0, 0), (0, 1, 0), (0, 0, 1)))
    try:
        while True:
            pose = car.pose  # Pose(x, y, angle), or None.
            if pose is None:
                car.stop()
            else:
                print(car.position, pose.angle, gps.balls)
                car.set_velocity(x=.2, y=0, w=0)
            time.sleep(.05)
    except KeyboardInterrupt:
        pass  # Leaving the with block stops and disconnects.
```

`GPS().cars` maps marker numbers to immutable `Pose` objects; `GPS().balls` is a
tuple of `Position(x, y)` objects (currently the detector finds one ball).
Coordinates are board-frame metres; `pose.angle` is degrees CCW from +X.
`car.wait_for_position()` waits up to 10 seconds for a fresh pose and explains
missing WiFi telemetry, an undetected board, or a missing marker. Reading
`car.pose` reports position loss/recovery once per transition, without printing
every loop iteration. The student `main.py` keeps these diagnostics in the backend.
`car.position` is `(x, y)` or `None`. Objects disappear immediately when omitted
from a packet, and feedback expires after **0.5 seconds**. When available,
server-reported frame age and excess delivery delay are included; buffered or
out-of-order packets do not renew a pose. Excess delay is relative to the fastest
packet observed, so this does not require synchronized clocks and cannot detect
a constant delay already present on the first packet. Read `car.pose` once when
you need a consistent pose.

For TE2004B broadcasts without ball data, use `GPS(port=5000, subscribe=False)`
before creating any cars. Missing ball fields are accepted as an empty collection.

Before creating any cars, optional `GPS(port=45888, stale_after=.5)` configures
the shared receiver. Later `GPS()` calls reuse it; conflicting settings raise
an error. There is one UDP socket per process. `gps.close()` releases it.
GPS automatically discovers the lab on UDP **45889**, renews a subscription
once per second, and receives feedback by unicast on **45888**. This avoids
WiFi broadcast loss, particularly while Bluetooth is active. No IP address or
extra student code is needed. Subscriptions expire after five seconds; closing
GPS unsubscribes immediately. Discovery requires the lab hotspot; Tailscale
does not forward discovery broadcasts. Stop any diagnostic listener already
using the feedback port. Custom telemetry ports use `port + 1` for discovery.

`Car(number)` waits for BLE readiness (default timeout 15 seconds), independently
of camera visibility. `car.is_connected` reports BLE status; `car.stop()` sends
an acknowledged zero command and `car.close()` stops/disconnects. Commands are
normalised **[-1, 1]**, in robot axes: x forward, y left, w CCW. They are not
metres/second and stay active until changed/stopped. Tracking loss does not
automatically stop a car: handle `pose is None` in your loop, as above.

`car.set_trim(matrix)` writes the single 3x3 correction matrix and returns its
readback. Rows are corrected x/y/w outputs; columns are requested x/y/w inputs:

```python
# Illustrative coefficients; tune for the actual robot on the track.
car.set_trim((
    (1.0,  0.0, 0.0),  # Corrected x
    (0.02, 1.0, 0.0),  # Corrected y: add 0.02 * requested x
    (-0.04,0.0, 1.0),  # Corrected w: add -0.04 * requested x
))
print(car.trim_matrix)
car.set_trim()  # Reset to identity: no gain or cross-axis correction.
```

The diagonal entries are the axis gains; no separate gain interface is needed.
All entries must be finite. Diagonal gains must be in **[0, 2]**; off-diagonal
corrections in **[-2, 2]**. Every call replaces the whole matrix. Gain-only
adjustment uses a diagonal matrix with zeros elsewhere.

The ESP32 multiplies the decoded command by this matrix before wheel mixing.
Forward input 0.5 in the example yields `(0.5, 0.01, -0.02)`. Reverse input
flips the correction signs; simultaneous inputs combine linearly. Zero input
always produces zero output. Because correction happens after BLE byte decoding,
small cross-axis corrections bypass the client's command rounding. Each corrected
axis is clipped to [-1, 1]; leave command headroom for effective compensation.
This linear feedforward correction can supplement closed-loop pose/heading control.

`trim_matrix` is a cached readback refreshed on reconnect/configuration, or
`None` on firmware without matrix support. Driving still works on that firmware,
but trim configuration requires updating it. Settings last across connections,
reset on reboot, and are not written to flash. Configure while stopped; motor
power can remain off. The former three-gain API and BLE characteristic are removed.

## Lab vision server

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r detection/requirements.txt
bash detection/run.sh --setup setup_160cm --headless
```

`detection/run.sh` checks the existing lab hotspot, obtains its broadcast address, and
launches camera capture, a browser viewer, and vision together. Without an
explicit `--setup`, it offers a board-size menu. Omit `--headless` to show the
camera and board windows on a graphical desktop. Ctrl+C stops owned processes.

For the lowest telemetry latency when neither the ball nor a camera preview is
needed, run the server as:

```bash
bash detection/run.sh --setup setup_160cm --headless --no-ball-detection --no-camera-viewer --rate 30
```

This skips ball processing and the HTTP/JPEG viewer and publishes every processed
frame (up to 30 Hz). Omit `--no-ball-detection` when ball positions are required;
omit `--no-camera-viewer` when the browser preview is required. These options do
not change the packet format: disabled ball detection simply publishes an empty
`balls` list.

The lab copy is `/home/aavila/Desktop/VSSS`, beside `TE2004B_AD26`. VSSS uses
its own `vsss-camera-<user>` frame buffer. Do not launch both projects against
the same physical camera simultaneously; the default viewer port is also 8080.

View the camera at `http://192.168.12.1:8080/` on the lab hotspot, or
`http://100.87.82.76:8080/` through Tailscale. The viewer displays the camera
feed; the local OpenCV windows add detection outlines and the board plot.

For direct capture without the viewer/supervisor:

```bash
python -m detection.main --setup setup_160cm --camera-source direct --headless --broadcast 192.168.12.255
```

## Position packets

```json
{
  "timestamp": 1794108208.396,
  "board_detected": true,
  "markers": [{"id": 2, "x": 0.435, "y": -0.037, "angle": 180.0}],
  "balls": [{"x": 0.454, "y": 0.048}],
  "frame_sequence": 166,
  "frame_age_ms": 112.575
}
```

GPS clients receive packets by unicast on UDP **45888**. Legacy broadcasts on
that port remain available for `telemetry_listener.py` (separate from TE2004B_AD26
on 5000). Coordinates are metres in the board
frame, +Y up; marker headings are degrees, positive counterclockwise from +X.
The default publication target is 10 Hz, limited by processing speed. Use
`--rate 30` to publish every processed frame when vision runs below 30 FPS.
`frame_age_ms` measures capture-return to packet generation on the server,
not end-to-end network delay. It is included when using the camera service.

Run this on a receiving computer on the hotspot:

```bash
python telemetry_listener.py
```

GPS handles subscription automatically; the ESP32 needs no WiFi connection. Autonomous controllers
should stop when the board/target disappears or packets become stale.

## Ball and marker calibration

`detection/ball_det.py` restores the HSV/circularity-based BallDetector from `main`.
`detection/ball_thresholds.txt` contains its lower/upper OpenCV HSV bounds; the current
values were measured from the blue lab ball. Restart vision after editing
these thresholds. The detector currently reports its best circular candidate.

`--ball-height 0.02` is the default ball-centre height above the board, matching
the old VSSS setup. Set it to the actual ball radius for parallax correction.
Balls are detected before coloured overlays and only get world coordinates
when the reference board is detected; otherwise `balls` is empty.

`detection/marker_config.json` supports per-ID `height_m` overrides and optional display
styles. Heights reload during operation. Flat markers should use height 0;
markers mounted on robots should use their measured mounting height. The
default robot marker height remains 0.12 m from the imported vision setup.

## Maintenance and tests

`detection/configure_lab_network.sh` checks/repairs the lab's existing create_ap hotspot;
it does not set up an arbitrary new network. `update_repo.sh` fast-forwards the
current Git branch without discarding local work or switching to `main`.

```bash
python -m unittest discover -s tests -v
```

The camera transport, marker tracking, and launchers were imported from
TE2004B_AD26. Tests use synthetic camera frames and fake BLE; no motors move.
