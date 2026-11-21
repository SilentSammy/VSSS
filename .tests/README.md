# Tests and robot experiments

`python -m unittest discover -s .tests -v` runs regression tests with simulated
robots and fake BLE. It never moves a physical car.

The provider tests cover shared car access, motion takeover, stop on owner exit,
shared UDP subscriptions across processes, original-byte fanout, malformed
packets, observer overflow, and GPS recovery after provider failure. Replay
tests use real bag/plot subprocesses and check shared timing, metadata, clean
completion, provider conflicts, and rejection of physical car control.
Plot tests require the optional dependencies: `pip install -e ".[plot]"`.
Simulation tests cover deterministic motion, polygon contacts, kicks/rolling,
wall and ball collisions, waypoint arrival/timeout, and configuration validation.
Real simulation subprocess checks exercise the public API, GPS, shared ownership,
teleop, observer bags, plotting, provider conflicts, and restart. Hardware source,
scanner, and worker paths are forbidden in these simulation subprocesses.
Bag-seeding checks cover current/legacy headers, metadata, overrides and ignored
tails. Editor tests cover IPC, exclusive grabs, abrupt disconnect recovery,
source binding, clamping and view-only replay. Mouse tests cover translation,
rotation, ball dragging, lost button release and renderer coordinate transforms.
Latency tests check delayed UDP/onboard feedback, BLE acknowledgement timing,
zero-delay operation and progressing physics during delayed requests. The
student ball-to-zone example has API/geometry/failure tests and a real simulator
subprocess trial with both latency and wobble enabled.
Controller regressions check forward feed at contact, continuous translation
during moderate heading correction, bounded lateral/yaw commands, earlier
release for the final coast, and waiting for a loose ball before replanning.
They also check moderate capture, the recorded bag-8 first push pose, continuous
feed across bag-12 centering thresholds, rotation locking after contact,
stale onboard pose diagnostics, glancing-contact rejection, outward-edge pushes
and stopping outside the field.
Bag-analysis tests check command/payload decoding, measured write durations
and raw outside detections using header geometry.
Kick-example tests check rectangular detour clearance, one-waypoint happy-path
sequencing, full straight commands without rotation, and stopping on feedback
loss/timeout or invalid approach geometry.
Scenario tests check two-frame velocities, car ID/angle matching, malformed
pairs, seeded goalie shots, saved starts, optional bag-motion loading and the
wait-for-first-command gate through both physics and a real CLI process.
Goalie tests check crossing prediction, ignored receding shots and X-only commands.

Robot experiments live in [tools/run.py](../tools/run.py).
`python -m tools.run --waypoint 0 .5 90` previews without connecting.
`--execute` enables motion. `--observe SECONDS` records pose and BLE status
without sending motion commands. Configuration writes require explicit JSON
arguments. Experiment commands are never executed during test discovery.
