# Previous experiment recordings

Historical recordings remain in `calibration/runs/` at their original paths.
Reusable experiment code now lives in [tests/experiments](../tests/experiments/README.md).

New commands use `python -m tests.experiments.run`. Preview is the default;
actual motion requires `--execute`. No tests move the physical robot unless
that explicit experiment command is used.
