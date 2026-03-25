"""
Proof of concept: one process sends data, another process owns the plot.

Uses PyQtGraph's built-in multiprocess module (QtProcess), which:
  - Spawns a child process with its own Qt event loop
  - Exposes remote objects via transparent proxies
  - _callSync='off' makes data pushes fire-and-forget (non-blocking)

Expected output: [main] lines should stay ~20ms regardless of plot activity.
"""
import time
import math
import pyqtgraph.multiprocess as mp
from pyqtgraph.Qt import QtWidgets
import sys

# A QApplication must exist in the main process before spawning a QtProcess
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

# Spawn child process with a Qt event loop
proc = mp.QtProcess()

# Import pyqtgraph *in the child process* and create a plot window there
rpg = proc._import('pyqtgraph')
win = rpg.plot(title='Remote Process Plot')
curve = win.plot(pen='y')

print("Child process running. Press Ctrl+C to quit.")
print()

t = 0.0
try:
    while True:
        t0 = time.monotonic()

        xs = [i * 0.05 for i in range(200)]
        ys = [math.sin(x + t) + 0.3 * math.sin(3 * x + t) for x in xs]

        # Push data to child process — _callSync='off' = non-blocking
        curve.setData(x=xs, y=ys, _callSync='off')

        elapsed = (time.monotonic() - t0) * 1000
        print(f"[main] {elapsed:.2f} ms")

        t += 0.1
        app.processEvents()  # keep local Qt event loop alive for IPC
        time.sleep(0.02)

except KeyboardInterrupt:
    pass
finally:
    proc.close()
