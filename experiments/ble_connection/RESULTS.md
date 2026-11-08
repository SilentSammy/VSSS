# BLE discovery, restart, and handoff experiments

Tested November 3, 2026 on Windows with the ESP32-C3 attached to COM17,
MicroPython 1.29.0, device name VSSS_LED_TEST, address 88:56:A6:62:7E:A2.
The test only controls GPIO8; motor modules are never imported.
Global Bleak is 1.0.1. Bleak 3.0.2 was separately installed into a temporary
directory for comparison; the global installation was not upgraded.

## The original 10-second bottleneck

`mecanum_client.py` previously did both of these:

1. `BleakScanner.discover(timeout=10)` collected all devices for ten seconds,
   even when the target advertisement had already arrived.
2. `BleakClient(device.address)` discarded the discovered BLEDevice and made
   Bleak do implicit address discovery again.

The old path took 12.44–12.88 seconds from starting discovery to completing
the first acknowledged LED write in three fresh Python processes. Median:
12.88 seconds. First advertisements arrived much earlier: 0.22–2.80 seconds.
These were unpaired connections, not Bluetooth pairing operations.

## Initial warm-cache comparison

Each sample launches a fresh Python interpreter, connects, writes LED-on,
reads it back, writes LED-off, explicitly disconnects, and exits. The ESP32
stays running between samples. Advertising interval is controlled over USB.
The 49 initial samples all succeeded.

Times below end at completion of the first acknowledged write. Python process
startup/import time is excluded; `process_wall_s` in the raw data includes the
whole process, readback, LED-off, and disconnect.

| Path | Advertising | Samples | Median ready | Maximum ready |
| --- | --- | --- | --- | --- |
| Original fixed scan + address | 100 ms | 3 | 12.881 s | 12.881 s |
| First name match + address | 100 ms | 5 | 0.846 s | 1.249 s |
| First name match + BLEDevice | 100 ms | 10 | 0.803 s | 1.440 s |
| Known address, implicit discovery | 100 ms | 5 | 0.733 s | 1.003 s |
| First match + BLEDevice + service filter | 100 ms | 5 | 1.105 s | 1.371 s |
| Service filter + forced Windows cache | 100 ms | 5 | 0.597 s | 0.808 s |
| First name match + BLEDevice | 20 ms | 8 | 0.679 s | 0.720 s |
| First match + BLEDevice + service filter | 20 ms | 8 | 0.548 s | 0.880 s |

This matrix is in `results.json`. Windows was already familiar with this
peripheral. These results are not a guarantee for a never-seen robot or a
different student's computer. Modes were interleaved within blocks, not
fully randomized. The two 100 ms name-device blocks bracket the 20 ms block.

## Why the fastest row is not the production default

After firmware restarts and repeated reconnections, service-filtered
enumeration became unreliable, returning Windows error -2147024874,
"The device does not recognize the command." Explicit caching did not fix it.
Unfiltered connections continued to work. The exact Windows/firmware-cache
interaction has not been isolated.

`handoff_filtered_results.json` preserves those failures. It also contains
same-process BLEDevice reuse results: four successful connections took
0.26–0.37 seconds, but one failed. This is insufficient to recommend that
pattern as a reliable optimization.

Following firmware resets/cache experiments, full GATT enumeration took
roughly 2 seconds and some discovery attempts took another roughly 2.5 seconds.
The initial sub-second numbers describe a favorable cache state, not every
subsequent connection.

The actual updated production client was therefore validated separately:

| Version | Successful fresh processes | Median connection-ready | Maximum |
| --- | --- | --- | --- |
| Global Bleak 1.0.1 | 5/5 | 4.591 s | 4.657 s |
| Isolated Bleak 3.0.2 | 5/5 | 4.534 s | 4.666 s |

The first sample in each suite was faster (2.36 s and 2.99 s respectively).
All ten samples additionally verified LED-on readback and LED-off writes.
There was no material upgrade benefit in this small comparison. Version and
native WinRT dependency changes were tested together, not separately.
See `client_results.json` and `client_latest_results.json`.

## Abrupt exit is a separate handoff problem

With no device-side lease, five clients successfully connected and then used
`os._exit(0)` to emulate an IDE stop/crash without BLE cleanup. All five
immediately launched replacement clients failed to discover the peripheral
within ten seconds. Because the ESP32 still considered the central connected,
it had not resumed advertising. This was not a pairing prompt or ten-second
fixed scan. See `handoff_results.json`.

An experimental device-side idle-disconnect lease of 1500 ms was then tested.
It starts after the first command; initial GATT setup gets a 10-second grace
period. Four clients completed a real abrupt exit, and all four replacements
connected and sent their first command in 2.94–3.05 seconds. The fifth attempt
failed before performing the abrupt exit; its later replacement connected,
but is not counted as a successful crash recovery. See
`handoff_watchdog_results.json`.

The lease exists only in `esp32_probe.py`, disabled unless USB command
`W 1500` enables it. It is NOT enabled in the production car server. The old
client suppresses repeated identical velocity writes, so production use of an
idle lease requires a heartbeat protocol first. Otherwise an intentionally
idle script or a constant-velocity command could be mistaken for a dead client.

All ownership tests above used separate processes on one Windows computer.
The other SSH-accessible computer exposed no Bluetooth controller and had
an inactive Bluetooth service, so a physical two-computer handoff was not
tested. Its configuration was not changed.

## Changes applied

- `mecanum_client.py`: stop scanning as soon as the name matches, retaining the
  ten-second maximum for an absent robot; pass BLEDevice directly to Bleak.
- Report readiness only after `await client.connect()` finishes GATT discovery.
  The old physical-link status could become true early, causing first-command
  failures and disconnect assertions. The integration checks exposed this race.
- Cancel an unfinished connection when `disconnect()` is called. Three real
  cancel-during-discovery checks returned in 2.6–14.9 ms; all remained disconnected.
- Clear cached velocity after connection/disconnection so the first identical
  command of a new connection is not skipped.
- `mp/ble_server.py`: use a configurable 20 ms advertising interval instead of
  100 ms, including when advertising resumes after disconnect. Battery impact
  was not measured.
- No forced service caching, service filtering, pairing, or package upgrade was
  enabled in production. The existing three-second retry delay remains for
  genuine connection failures; successful discovery no longer waits ten seconds.

`client_before_results.json` preserves the readiness-race integration failures.
`shutdown_results.json` records the cancellation checks. Syntax and diff checks
passed. The unrelated working changes to `mp/mecanum.py` were not edited.

## Implications for the overhaul

For normal scripts, always use `try/finally` or a context manager to disconnect.
In the initial matrix, explicit disconnect took about 0.11 seconds. It releases
the robot for another owner rather than relying on Windows after a process dies.

For robust crash recovery, add a heartbeat and a short robot-side ownership
lease, with motor stop before releasing the connection. The experiment supports
this direction but does not yet implement the complete car protocol.

For repeated script restarts on the same computer, a persistent BLE owner
process with local IPC would avoid reconnecting on every run entirely. That
architecture was not implemented or benchmarked here.

## Reproduce

- `python experiments/ble_connection/benchmark.py`
- `python experiments/ble_connection/handoff.py`
- `python experiments/ble_connection/handoff.py --watchdog-ms 1500`
- `python experiments/ble_connection/client_smoke.py`

These scripts assume COM17 and the test name/address above. They run the device
test in RAM and require `ble_server.py` on its filesystem. Normal completion
stops advertising and leaves GPIO8 off. Following this session, BLE and WiFi
were disabled and COM17 released. No boot/main file was replaced.

## Primary sources consulted

- [Bleak client: BLEDevice versus implicit address discovery, pairing, cleanup](https://bleak.readthedocs.io/en/latest/api/client.html)
- [Bleak scanner and first-match discovery](https://bleak.readthedocs.io/en/latest/api/scanner.html)
- [Windows service cache options and changing DIY GATT layouts](https://bleak.readthedocs.io/en/latest/api/args.html)
- [Bleak usage: explicit disconnection and cached OS state](https://bleak.readthedocs.io/en/latest/usage.html)
- [Maintainer discussion of Windows keeping a connection after process exit](https://github.com/hbldh/bleak/discussions/1276)
- [MicroPython advertising interval and disconnect events](https://docs.micropython.org/en/latest/library/bluetooth.html)

Latest online docs describe newer Bleak versions; API availability was also
checked against the installed Bleak 1.0.1 source before using it.
