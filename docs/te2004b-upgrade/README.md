# Consolidating pose telemetry into TE2004B_AD26

## Decision

Make `TE2004B_AD26` the single source of truth for camera capture, marker pose
estimation, optional ball detection, plotting, and UDP telemetry.

VSSS should consume the service over its documented packet protocol. Once the
features below are merged and validated in TE2004B_AD26, remove VSSS's copied
`detection/` implementation instead of maintaining another fork.

This is a migration plan, not evidence that the copies can be deleted today.

## Why TE2004B should own it

TE2004B_AD26 is already the more general camera/pose project. It contains the
camera service, board setups, marker configuration, viewer, benchmarks, Arduino
consumers, and UDP examples. VSSS added robot-specific requirements to a copy of
that stack. Those requirements are useful to TE2004B too and should move upstream.

The desired boundary is:

```text
TE2004B_AD26
  camera -> board/marker/ball estimation -> plot/viewer -> telemetry service

VSSS, ESP32 robots, Arduino examples, student scripts
  telemetry clients only
```

No VSSS motor, BLE, calibration, or controller code belongs in TE2004B_AD26.

## Proposed process architecture

The consolidation is also an opportunity to protect pose estimation from optional
ball and visualization work. Do not create a process for every small function;
separate work according to latency and failure requirements.

```text
run.sh / Python supervisor
|
+-- Camera process
|     Captures continuously into a sequence-keyed shared-memory ring
|
+-- Pose process (latency-critical, highest priority)
|     Reads the newest frame
|     Estimates board and marker poses
|     Publishes compact pose telemetry immediately
|     Sends pixel-space marker geometry over local IPC
|
+-- Ball process (optional, best effort)
|     May skip frames and never delays pose publication
|     Detects ball pixels and obtains matching board projection data
|     Publishes compact ball state and local drawing geometry
|
`-- Visualization process (optional, best effort)
      Retrieves the matching raw frame from the ring
      Draws local marker/ball annotations
      Renders the top-down field plot
      Owns OpenCV windows and GUI event handling
```

`run.sh` should still be the single operator entry point. A supervisor should start,
monitor, and stop all owned children together. Optional workers must not prevent
headless pose service startup.

### What should remain together

Keep pose estimation and pose UDP publication in the same process. Encoding a
small JSON packet and calling `sendto()` is inexpensive. A separate broadcaster
would add IPC, queueing, synchronization, and another failure mode without removing
meaningful computation from the critical path.

The camera service and JPEG viewer are already substantially separated. Preserve
that design and move local OpenCV visualization out of the pose loop as the first
process-isolation improvement.

### Local visualization data is not public telemetry

The visualization process needs image-space geometry, not merely field poses:

- marker IDs and four pixel corners;
- detected/tracked state;
- optional board outline pixels;
- ball centre/radius or contour pixels;
- `stream_id` and `frame_sequence` for every result.

This information must use local IPC on the vision computer. Never add marker
corners, ball contours, or full annotated images to the WiFi broadcast/unicast
packet. Doing so wastes airtime and can push UDP datagrams beyond the usual network
MTU, causing fragmentation and making broadcast loss substantially worse.

The channels have deliberately different schemas:

```text
Local IPC:
  raw frame ring + pixel corners/contours + drawing metadata

Network telemetry:
  marker ID/x/y/angle + ball x/y + stream/frame/timing metadata
```

Reasonable local result transports include bounded multiprocessing queues,
Unix-domain sockets, or small shared-memory latest-result slots. Localhost UDP is
acceptable if independent processes are valuable, but it is not the public
broadcast and must remain machine-local. Avoid adding ZeroMQ or another dependency
unless the standard-library implementation becomes demonstrably fragile.

### Matching annotations to frames

A single latest-frame slot is insufficient for reliable camera overlays: by the
time pose detection finishes, capture may have replaced that frame. Use a bounded
shared-memory ring of approximately 3-5 raw frames keyed by `stream_id` and
`frame_sequence`.

At 1920 x 1080 BGR, one frame is approximately 6.2 MB; five frames consume roughly
31 MB plus metadata and synchronization. This is reasonable on the lab desktop and
avoids repeatedly copying full images between processes.

Detection workers publish lightweight geometry with the source sequence. The
visualizer retrieves exactly that frame, draws the geometry, and discards results
whose frame has already left the ring. It must never make a detection worker wait.

An easier but less desirable alternative is drawing annotations in the pose worker
and transferring annotated frames. That keeps synchronization simple but leaves
drawing on the critical path and copies large images between processes.

### Coordinating ball localization

Ball segmentation produces an image-space centroid, but field coordinates require
the board transform from the corresponding frame. Do not repeat board estimation
inside the ball worker merely to make it independent.

Preferred design:

1. Pose worker retains a short sequence-keyed ring of board transforms.
2. Ball worker returns pixel centroid/contour plus stream/frame identity.
3. Pose-side aggregation matches the result to the retained transform and projects
   the ball into field coordinates.
4. Pose publication never waits for a ball result.

Alternatively, the pose worker can publish transforms locally for the ball worker
to consume, but both sides must enforce exact stream/frame matching.

### One network port or two

Both designs are viable:

- One port: pose packets are sent immediately; later packets may update optional
  ball state. Consumers distinguish message type and freshness.
- Two ports: pose remains on the established port and balls use a best-effort port
  with their own rate.

Two ports isolate rates but make client-side fusion more complex. If used, every
ball packet needs `schema_version`, `stream_id`, `frame_sequence`, `frame_age_ms`,
timestamp, and `board_detected`. The student-facing `GPS` API should fuse the feeds
and expose only fresh data.

Do not delay a current pose packet while waiting for a ball result under either
design. Do not retransmit old poses to fill sequence gaps.

### Is the extra complexity justified?

Process isolation is most likely worthwhile for visualization. Earlier VSSS tests
observed roughly 11-12 Hz with UI work versus 13-14 Hz in an optimized headless
configuration, suggesting approximately 15-25% throughput loss plus potentially
more important long-tail frame gaps.

Add the ball worker only after measuring ball detection separately. Keep it if it
provides at least one material result, such as:

- about 20% higher pose publication frequency while balls are enabled;
- materially lower p95/p99 frame age or update interval;
- substantially fewer frame-sequence gaps;
- elimination of intermittent robot zigzag with balls/UI enabled.

If measurements do not show a meaningful improvement, retain ball detection in the
pose process behind `--no-ball-detection`. The goal is a predictable high-priority
pose pipeline, not the largest possible process count.

## Features to upstream

### 1. Reliable unicast subscriptions

Keep UDP broadcast for discovery and legacy consumers, but add expiring unicast
subscribers for normal feedback delivery.

The implementation already proven in VSSS is `detection/telemetry.py`:

- data port: configurable `PORT`;
- discovery/subscription port: `PORT + 1`;
- `VSSS_SUBSCRIBE` registers the sender's address for five seconds;
- clients renew once per second;
- `VSSS_UNSUBSCRIBE` removes a client immediately;
- subscriber work per camera frame is bounded;
- at most 256 subscribers are retained;
- one dead client cannot stop publication;
- broadcast continues when `--broadcast` is supplied.

Rename the messages to protocol-neutral names before making this canonical, for
example `POSE_SUBSCRIBE` and `POSE_UNSUBSCRIBE`. During migration, accepting both
the old and new strings is inexpensive and avoids a synchronized deployment.

Why this matters: WiFi broadcast frames are normally unacknowledged. Unicast gets
link-layer acknowledgement/retry and should reduce intermittent frame loss,
especially on an ESP32-C3 using WiFi and BLE simultaneously.

Do not hard-code the VSSS port 45888 into TE2004B. Preserve TE2004B's current
default port 5000 unless there is a deliberate compatibility break. VSSS clients
and robot firmware should make the server/port configurable.

### 2. Optional ball detection

Port these VSSS files/features:

- `detection/ball_det.py`;
- `detection/ball_thresholds.txt`;
- ball localization through `BoardEstimator.project_point_to_board`;
- `--ball-height` for parallax correction;
- `--no-ball-detection` to remove its CPU cost;
- ball drawing in the camera image and top-down plot;
- a `balls` array in telemetry.

Ball detection must run before coloured marker/plot overlays are drawn so those
overlays cannot become false ball candidates.

Packet compatibility rule:

- enabled and detected: `"balls": [{"x": ..., "y": ...}]`;
- enabled but absent: `"balls": []`;
- disabled: preferably still `"balls": []`;
- older packets without `balls` remain valid to consumers and mean no balls.

The current detector returns the best circular HSV candidate. It is not a general
multi-ball tracker. Preserve that limitation in the public documentation.

### 3. Steady telemetry scheduling

TE2004B currently checks elapsed time from the previous send. When processing is
near the requested interval, small scheduling jitter can make a nominal 10 Hz
stream send every second frame, effectively approaching 5 Hz.

Use a deadline-based schedule:

```python
next_telemetry = time.monotonic() + interval

if now >= next_telemetry:
    publish(packet)
    next_telemetry = max(next_telemetry + interval, now)
```

This change is independent of balls and unicast and should be upstreamed even if
the rest of the migration is postponed.

### 4. Low-latency/headless operation

Port the runtime switches that let the operator remove optional work:

- `--headless`: no local OpenCV windows;
- `--no-ball-detection`: marker/pose processing only;
- `--no-camera-viewer`: no HTTP/JPEG encoding process;
- `--rate 30`: publish every processed frame up to the requested limit.

The useful lab command should remain one launcher invocation, for example:

```bash
bash run.sh --setup setup_160cm --headless \
  --no-ball-detection --no-camera-viewer --rate 30
```

Disabling optional work must not change the packet schema. It should only produce
an empty `balls` collection and omit UI/viewer overhead.

### 5. Ball-aware board plotting

Extend `BoardPlotter2D.render` with optional ball centres and a physical radius.
The VSSS copy already handles the Cartesian `+Y up` to image `+Y down` conversion.
Keep `render(markers)` working for existing TE2004B callers.

### 6. Stream identity and timing metadata

TE2004B already emits `frame_sequence` and `frame_age_ms` when frames come from
the camera service. Preserve them and make their semantics part of the protocol:

- `frame_sequence`: source-frame sequence, increasing within one stream;
- `frame_age_ms`: capture-return to packet-generation age on the vision computer;
- `timestamp`: wall-clock packet-generation time, not safe for latency calculation
  unless clocks are synchronized.

Add a `stream_id` generated when capture starts. A client can then distinguish a
camera restart from packet reordering or a wrapped/reset sequence number.

If practical, give direct-capture mode the same sequence/age fields rather than
only providing them through the shared camera service. Consumers should still
tolerate missing timing fields during migration.

These fields are essential for diagnosing the VSSS robot's intermittent zigzag:
controllers need to know whether frames were delayed, dropped, duplicated, or
generated after a source restart.

### 7. Protocol versioning

Add a small integer `schema_version` to packets when the canonical format is
settled. This avoids detecting future protocol changes from field presence.

Recommended initial canonical packet:

```json
{
  "schema_version": 1,
  "stream_id": "random-per-camera-start",
  "timestamp": 1794108208.396,
  "frame_sequence": 166,
  "frame_age_ms": 38.2,
  "board_detected": true,
  "markers": [
    {"id": 1, "x": 0.125, "y": 0.742, "angle": 91.0}
  ],
  "balls": [
    {"x": 0.454, "y": 0.048}
  ]
}
```

Unknown fields must be ignored. Optional collections must default to empty.

### 8. Tests and observability

Bring over or recreate tests for:

- subscribe, renewal, expiry, unsubscribe, and subscriber limits;
- simultaneous broadcast and unicast delivery;
- malformed discovery traffic not interrupting publication;
- a closed/failed subscriber not affecting other clients;
- packet compatibility with missing/empty/present balls;
- ball projection and `--ball-height` validation;
- `--no-ball-detection` avoiding detector construction;
- top-down ball rendering;
- steady-rate scheduling around the requested interval;
- frame sequence, age, and stream restart semantics;
- headless/no-viewer launch combinations.

Add optional periodic telemetry diagnostics rather than printing every packet:

- processed FPS and publication Hz;
- active unicast subscriber count;
- latest frame age;
- sequence gaps visible at a diagnostic receiver;
- ball-processing time when enabled.

The service should not add per-client acknowledgement at the application layer;
WiFi unicast acknowledgement plus sequence-gap measurement is sufficient for this
real-time feed. Retransmitting old poses can be worse than dropping them.

## Changes that are not VSSS features

Some differences between VSSS `detection/` and TE2004B are packaging details,
not upgrades to copy upstream:

- VSSS uses relative imports because detection is a subpackage.
- VSSS resolves resources relative to `detection/__file__`.
- VSSS activates the repository-root virtual environment from a nested launcher.

If TE2004B remains a flat root application, retain its simpler imports and paths.
Only adopt package layout changes if TE2004B is intentionally converted into an
installable package.

The VSSS copy also contains Windows process-tree cleanup improvements and handles
`ConnectionAbortedError` for viewer clients. These are sensible robustness fixes,
but they are separate from the telemetry consolidation and should receive their
own small tests/commit.

## Consumer migrations

### VSSS laptop client

`gps.py` already implements expiring subscription renewal and accepts a missing
`balls` field. Change its default port only if TE2004B deliberately changes the
canonical port. Keep freshness, frame-age, and out-of-order rejection.

### ESP32 robot

The current firmware uses passive broadcast reception:

```python
PoseReceiver(car_id, subscribe=False)
```

Change it to subscription mode after the TE2004B publisher is deployed. Make the
server/discovery address and port part of `robot_config.json`, rather than relying
on VSSS constants. The ESP32 should:

- renew its subscription once per second;
- discard poses for other marker IDs after parsing;
- update control once per new frame sequence;
- stop on stale pose;
- count sequence gaps/restarts for diagnostics;
- never replay delayed poses to "fill" a gap.

### Arduino and legacy listeners

Keep broadcast enabled during migration. Existing consumers should continue to
read `board_detected` and `markers` while ignoring new fields. Update examples to
show subscription mode where the platform supports it.

## Repository ownership after migration

TE2004B_AD26 should contain:

- camera/frame service;
- board and marker estimation;
- optional ball detector;
- plotter/viewer;
- telemetry publisher and protocol documentation;
- board/marker resources;
- vision/telemetry tests and benchmarks.

VSSS should contain:

- `gps.py` protocol client;
- ESP32 `pose_receiver.py` protocol client;
- BLE, motors, control, calibration, and student APIs;
- robot experiments and robot-specific tests.

VSSS should not import TE2004B source at runtime. The lab runs TE2004B as a
separate service. If VSSS developers need a local vision server, document cloning
and launching TE2004B next to VSSS instead of copying its modules.

## Migration sequence

1. Create a TE2004B feature branch; preserve its current default port and broadcast.
2. Add timing benchmarks for the current combined loop: mean/p95/p99 frame age,
   publication interval, and pose rate with UI/balls independently enabled.
3. Add the standalone telemetry publisher and its unit tests.
4. Replace direct `socket.sendto` in `main.py` with that publisher.
5. Fix deadline-based telemetry scheduling.
6. Move OpenCV visualization into an optional process using sequence-keyed local
   geometry; validate the frame ring before changing ball processing.
7. Add ball detection/localization, packet field, plot rendering, and tests.
8. Benchmark in-process balls against an optional best-effort ball worker. Keep the
   worker only if the acceptance threshold above is met.
9. Add `--no-ball-detection` and `--no-camera-viewer` performance switches.
10. Add `stream_id`, document timing fields, and update listeners.
11. Test broadcast and unicast simultaneously on the lab hotspot.
12. Switch one ESP32 to subscription mode and compare sequence gaps, update jitter,
   stale faults, and waypoint smoothness against broadcast.
13. Update the lab desktop checkout and service/launch documentation.
14. Point all VSSS documentation at TE2004B as the vision service.
15. Only then delete VSSS `detection/` and its duplicated resources/tests.

Keep the changes reviewable. Unicast, scheduler, balls, performance switches, and
repository deletion should not be one untestable commit.

## Acceptance criteria before deleting VSSS `detection/`

- Marker coordinates/headings match the current VSSS service on recorded frames.
- Ball output and plot can be enabled or disabled from the launcher.
- Disabled balls always produce a compatible empty collection.
- Legacy broadcast listeners continue working on port 5000 (or a documented,
  deliberately migrated port).
- Multiple unicast clients receive current frames and expire cleanly.
- ESP32 subscription survives at least a representative BLE + WiFi waypoint run.
- Publication rate does not halve near the requested processing frequency.
- Packets expose enough identity/timing data to diagnose gaps and restarts.
- Headless/no-viewer operation runs without GUI or JPEG work.
- Visualization uses local sequence-matched metadata and never enlarges public
  telemetry with corners, contours, or images.
- Slow visualization or ball processing cannot block pose publication.
- The process supervisor stops owned children cleanly and reports a failed critical
  pose/camera worker distinctly from an optional UI/ball worker.
- TE2004B tests and VSSS consumer tests pass.
- The lab launch command and viewer URL are documented and verified.
- No VSSS production import references the copied `detection/` package.

## Source comparison used for this plan

Compared on 2026-11-10:

- `C:\Users\SAMMY_LOQ\Desktop\Python\TE2004B_AD26`
- `C:\Users\SAMMY_LOQ\Desktop\Python\VSSS\detection`

The relevant proven VSSS additions are concentrated in `telemetry.py`,
`ball_det.py`, `ball_thresholds.txt`, `main.py`, `plotter2d.py`, and the
no-viewer launcher path in `camera_stream.py`.
