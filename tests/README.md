# Tests and robot experiments

`python -m unittest discover -s tests -v` runs regression tests with simulated
robots and fake BLE. It never moves a physical car.

Reusable robot experiment code lives in [experiments/](experiments/README.md).
`python -m tests.experiments.run` previews a plan without connecting to anything.
`--observe SECONDS` records telemetry without BLE. Only `--execute` enables a
physical experiment; it is not used during test discovery.
