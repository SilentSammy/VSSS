# Robot experiment setup

This package prepares repeatable car experiments. Preview is the default: it
opens no telemetry socket, connects to no BLE device and sends no commands.
Physical experiments require the explicit `--execute` option. Unit-test discovery
uses fake robots and never invokes a physical experiment.

Run commands from the repository root with the root Python requirements installed.

## Check setup without moving

```bash
python -m tests.experiments.run
python -m tests.experiments.run --experiment waypoints --controller pid
python -m tests.experiments.run --observe 5
```

The first two commands print configuration and plans. The last records telemetry
only: it never constructs `Car`, reads/writes trim, or connects to BLE. Inputs in
its CSV are **blank/unknown**, not fabricated zero commands; `command_owned` is
false. This distinction matters if something else is controlling the car.

The default source is **TE2004B**: UDP **5000**, passive broadcast reception,
marker positions and degree headings. Missing `balls` means an empty collection,
not an invalid packet. No ball is required and no VSSS subscription requests are
sent. For a custom server port, add `--telemetry-port PORT`; it must match the
server. For the VSSS unicast service use `--telemetry vsss` (default 45888).
Configure `GPS(port=5000, subscribe=False)` before constructing any student Cars
when using TE2004B outside this runner.

If the telemetry port is occupied, release the conflicting receiver or configure
a matching free server/client port. The runner does not stop other applications.
A preview does not require a free port or network access. Passive readiness
results are saved even if no packets arrive.

## Reusable components

- `session.py`: guarded command loop, recorded phases, target acquisition,
  recentering, stop acknowledgements, STOP-file and disconnect handling.
- `recording.py`: queued, flushed CSV recording of parsed packets from the
  selected transport and control events, including rests and recentering.
- `safety.py`: rectangular/radial bounds and predictive inner-boundary checks.
- `controllers.py`: measured predictive controller plus P and original pose-PID
  baselines. All expose `reset()` and `go_to(x,y,theta,tx,ty,ttheta)`.
- `plans.py`: immutable single-axis, segmented multi-axis and waypoint plans.
- `scenarios.py`: stride, matrix-correction comparison and waypoint experiments.
- `summarize.py` / `assess.py`: descriptive responses and curated data selections.

Controllers consume metre positions and radian headings internally. CSV and
waypoint JSON headings are **degrees CCW**. Predictive control estimates filtered
velocity from fresh camera frames, projects pose by 0.20 s, then applies bounded
P control to that predicted pose. Car 1's selected parameters and evidence are
saved in `calibration/controller_car1.json`; P and PID remain baselines.

## Prepared scenarios (preview only)

```bash
python -m tests.experiments.run --profile dynamic
python -m tests.experiments.run --experiment correction --trim-file tests/experiments/configs/correction_seed.json --bounds-x .8 --bounds-y 1.5
python -m tests.experiments.run --experiment waypoints --controller p
python -m tests.experiments.run --experiment waypoints --controller pid
```

### Correction comparison

The same pulse plan runs first with the identity matrix and then with the
candidate matrix supplied in `--trim-file`. The car is stopped before changing
matrix, then recentered. Every row includes the complete active matrix; events
label configuration changes. The example matrix contains a **provisional**
forward-to-yaw term from earlier data, not a calibration established today.
After the comparison the runner restores the original matrix when BLE is still
available. Restoration failures/disconnections are recorded. Check configuration
before a later run if connection loss prevented restoration.

The correction file can be a 3x3 JSON array or an object containing `matrix`.
It uses the same finite/range validation as `Car.set_trim` without connecting to
hardware. Correction does not run unless `--execute` is supplied.

### Strides

`quick` has six signed-axis pulses, 0.5 command for 0.75 s. `standard` sweeps
0.3, 0.5 and 0.7. `dynamic` has ten longer strides with combined commands and
axes entering/leaving mid-stride, including reversals without a stop. Dynamic
segments use their own durations, scaled with `--stride-scale`.

`--recenter-every` selects the recenter interval (dynamic default 2, otherwise 1).
The initial heading is preserved unless `--heading` is supplied. Every recenter,
rest, pulse and abort is recorded. `--start-trial` resumes a plan in a new output
directory with original trial numbers; use the same plan and heading arguments.

### Waypoints

The default `configs/waypoints.json` is a small rectangle. Supply `--waypoints`
for another JSON list of `[x, y]` or `[x, y, heading_degrees]` targets. An omitted
heading retains the home heading. Targets must fit inside the configured inner
bounds. Waypoint moves, rests and periodic recentering share the same recorder,
controller interface, limits and timeouts as stride runs.

## When explicitly authorized to move

Add **`--execute`** to a reviewed command above. Do not use it for passive setup.
Ctrl+C or an empty `STOP` file in the recording directory requests a stop. The
control loop normally checks every 50 ms; BLE acknowledgement takes additional
time. The firmware heartbeat timeout is the final fallback on link loss.

The default freshness limit is 0.5 s. Lost/stale position, lost BLE, a writer
failure, a target/recenter timeout (45 s), or total run timeout (600 s) aborts and
retains partial data. Dynamic and waypoint defaults use absolute bounds
**?0.8 m X, ?1.5 m Y**, with a 0.2 m margin and 0.6 s predictive horizon for
stride cutoffs. Compact quick/standard strides retain the 0.35 m radius unless
both `--bounds-x` and `--bounds-y` are specified. Future comparisons should use
limits appropriate to the track and robot footprint. This is not collision
avoidance, and physical overshoot is possible between camera updates.

## Data and interpretation

New recordings are in `tests/experiments/runs/` (ignored by Git). Historical
`calibration/runs/` files remain at their original paths.

- `metadata.json`: mode, telemetry source/port, complete plan and controller
  parameters, original matrix, targets, command resolution and clock information.
- `samples.csv`: telemetry, requested/quantized commands, target, controller,
  active matrix, BLE sequence/acknowledgement, phases and faults.
- `status.json`: progress and final status; `result.json`: outcome.

Use `event=telemetry` for real pose samples. Control rows repeat the latest pose.
`req_x/y/w` are desired commands; `input_x/y/w` are queued signed-byte setpoints
before firmware correction, not physical velocity or PWM measurements. Held
commands are repeated as heartbeats without being new input transitions.
`ack_sequence` is the latest acknowledgement observed, not an MCU actuation time.
`stop_ack` follows the acknowledged stop call. Missing/stale poses are flagged.

`t_s` and `received_t_s` share a local monotonic clock. Server timestamp, frame
sequence and server-side frame age are separate fields. GPS compares changes in
the sender timestamp with local monotonic time to detect additional buffering
relative to the fastest packet observed from that sender; it does not require
the clocks to be synchronized. `network_excess_delay_s` is that relative excess,
not absolute network latency. `minimum_pose_age_s` adds it to server frame age
and local time since receipt. Packets already older than the freshness limit or
out of order are still logged with `packet_usable=False` and `pose_valid=False`,
but do not refresh the pose used by controllers. Sender-clock jumps can cause
conservative rejection; restarting the receiver re-establishes the reference.
If source timestamps or frame age are absent, those components cannot be
checked. In particular, a constant delay present from the first packet is not
detectable by this relative comparison. Passive captures have unknown commands
and no BLE state or trim readback.

The passive readiness result counts every received packet, includes trailing
silence in the largest gap, and is false if a gap exceeded the freshness limit,
the car disappeared after acquisition, or packets were unusable. A passing
short observation is evidence for that time window, not a guarantee of future
network performance.

```bash
python -m tests.experiments.summarize "tests/experiments/runs/*/samples.csv"
python -m tests.experiments.assess calibration/runs/assessment/selection.json
```

The descriptive summary splits multi-axis strides into segments and labels
boundary cutoffs. Curated assessment leaves originals unchanged. Effective gains
integrate response through the following rest and divide by command exposure;
these are not validated full dynamics or final PID gains.

The default experiment center is **(0, 0.75 m)** so calibration runs use the
upper half of the track. `--experiment correction --auto-fit` measures six
signed identity-matrix pulses, fits a least-squares 3x3 decoupling matrix, and
validates it with the same six pulses. Add `--keep-trim` only when the validation
should remain active; aborted runs always restore the prior matrix. Outside an
active pulse, a short telemetry interruption causes a confirmed stop and a
bounded fresh-pose recovery. During an active pulse it still aborts.

Accepted per-car matrices live in `calibration/cars.json`. The student-facing
`Car(number)` applies the matching matrix after BLE connection, transparently
restoring calibration after a robot reboot.
