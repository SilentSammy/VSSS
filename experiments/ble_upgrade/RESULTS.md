# BLE upgrade bench verification

November 3, 2026. ESP32-C3 on COM17, MicroPython 1.29.0, advertising as Eq4.
The updated `main.py`, `ble_server.py`, and `mecanum.py` are installed and
automatic startup after reset was verified. The board was unmounted, so these
checks validate command handling and PWM allocation, not physical driving.

- All 12 process, heartbeat, firmware-timeout, and PWM regression tests passed
  on Windows/Python 3.11 and Ubuntu/Python 3.12. These use a simulated BLE radio;
  actual BLE hardware was tested from Windows only.
- Three final live connect/control/disconnect cycles passed. Connection-ready
  times, including worker startup: 3.34, 3.72, and 3.86 seconds. Each held a
  nonzero mixed-axis command for 2.5 seconds without expiring the lease.
- Normal disconnect took 0.23–0.25 seconds.
- A direct client deliberately sending no heartbeat was disconnected by the
  firmware after 1.61 seconds. Discovery then took 0.27 seconds.
- Following abrupt student-script exit, discovery succeeded after 1.81 seconds.
- Following forced termination of the BLE worker, discovery succeeded after
  1.47 seconds. These are rediscovery timings, not complete reconnect timings.
- The board completed 100 mixed-axis commands at 20 ms intervals after the PWM
  allocation fix. Motors are updated atomically and old PWM allocations are
  released before requesting a different frequency.

GATT registration followed immediately by advertising was unreliable on this
board during serial soft-reset experiments. A one-second startup allowance
produced a visible advertisement with the registered service and the full
robot program. This delay only applies when firmware starts, not each time
a student script connects. The radio is also reset before rebuilding GATT.

Raw timings are in `hardware_results.json`. `hardware_check.py` reproduces the
live checks but deliberately sends nonzero velocity commands.

Original board files are preserved in `device_backup/`. The motor pin changes
that were already present locally were preserved. Final zero velocity was
acknowledged, the client disconnected, and COM17 was released. The board stays
running and advertising as Eq4 for the user's robot test.
