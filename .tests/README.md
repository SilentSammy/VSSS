# Tests and robot experiments

`python -m unittest discover -s .tests -v` runs regression tests with simulated
robots and fake BLE. It never moves a physical car.

The provider tests cover exclusive car control, stop on controller exit,
shared UDP subscriptions across processes, original-byte fanout, malformed
packets, observer overflow, and GPS recovery after provider failure. Replay
tests use real bag/plot subprocesses and check shared timing, metadata, clean
completion, provider conflicts, and rejection of physical car control.
Plot tests require the optional dependencies: `pip install -e ".[plot]"`.

Robot experiments live in [tools/run.py](../tools/run.py).
`python -m tools.run --waypoint 0 .5 90` previews without connecting.
`--execute` enables motion. `--observe SECONDS` records pose and BLE status
without sending motion commands. Configuration writes require explicit JSON
arguments. Experiment commands are never executed during test discovery.
