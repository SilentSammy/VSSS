"""
Proof of concept: click events + external overlay artists on a remote plot.

Proves:
  1. File-based click IPC: child writes (x,y) to a temp file; parent polls mtime.
  2. External overlay: caller creates a remote PlotDataItem via win.plot() and
     holds the ObjectProxy — updates are fire-and-forget, no special framework
     needed. This is how GamePlotter2D consumers will draw custom shapes.

Click anywhere — (x,y) prints in terminal.
Watch the dashed cyan circle: drawn entirely by "external consumer" code below.
[main] timing should stay ~20ms throughout.
"""
import time
import math
import os
import tempfile
import pyqtgraph.multiprocess as mp
from pyqtgraph.Qt import QtWidgets
import sys

# A QApplication must exist in the main process before spawning a QtProcess
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

# Spawn child process with a Qt event loop
proc = mp.QtProcess()

# Import pyqtgraph *in the child process* and create a plot window there
rpg = proc._import('pyqtgraph')
win = rpg.plot(title='Remote Process Plot — click me!')
curve = win.plot(pen='y')

# --- File-based click IPC ---
click_file = os.path.join(tempfile.gettempdir(), 'vss_click.txt')
_last_mtime = 0.0

click_store = proc._import('_click_store')
click_store.setup(win, click_file)

# ---------------------------------------------------------------------------
# External overlay PoC
# ---------------------------------------------------------------------------
# Simulate a "consumer" of the plotter creating its own drawing.
# win.plot() returns an ObjectProxy to a PlotDataItem living in the child.
# The consumer holds this proxy and calls setData(_callSync='off') each frame.
# This is the entire overlay mechanism — no special framework required.

# A dashed cyan circle (parametric)
overlay_circle = win.plot(pen=rpg.mkPen('c', width=2, style=rpg.QtCore.Qt.DashLine))

# A magenta cross marker at the origin
overlay_marker = win.plot(
    x=[0], y=[0],
    pen=None,
    symbol='+',
    symbolSize=20,
    symbolPen=rpg.mkPen('m', width=2),
)

def _make_circle(cx, cy, r, n=64):
    angles = [2 * math.pi * i / n for i in range(n + 1)]
    return [cx + r * math.cos(a) for a in angles], [cy + r * math.sin(a) for a in angles]

print("Child process running. Click the plot to test click IPC. Ctrl+C to quit.")
print(f"Click file: {click_file}")
print()

t = 0.0
try:
    while True:
        t0 = time.monotonic()

        # Main curve
        xs = [i * 0.05 for i in range(200)]
        ys = [math.sin(x + t) + 0.3 * math.sin(3 * x + t) for x in xs]
        curve.setData(x=xs, y=ys, _callSync='off')

        # External overlay: pulsing circle driven by consumer code
        r = 1.0 + 0.5 * math.sin(t)
        cxs, cys = _make_circle(0, 0, r)
        overlay_circle.setData(x=cxs, y=cys, _callSync='off')

        # External overlay: marker drifting along x axis
        overlay_marker.setData(x=[math.sin(t) * 3], y=[0], _callSync='off')

        # Poll click file
        try:
            mtime = os.path.getmtime(click_file)
            if mtime != _last_mtime:
                _last_mtime = mtime
                with open(click_file) as f:
                    line = f.read().strip()
                if line:
                    cx, cy = map(float, line.split(','))
                    print(f"  [click] ({cx:.3f}, {cy:.3f})")
        except (FileNotFoundError, ValueError):
            pass

        elapsed = (time.monotonic() - t0) * 1000
        print(f"[main] {elapsed:.2f} ms")

        t += 0.05
        app.processEvents()  # keep local Qt event loop alive for IPC
        time.sleep(0.02)

except KeyboardInterrupt:
    pass
finally:
    proc.close()
