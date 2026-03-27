"""
Proof of concept: background image on remote plot, scaled to physical dimensions.

Proves that an image can be loaded, sent to the child process, and displayed
as a background with correct physical scaling (in metres).

Image: resources/gridboard_square.png
Target size: 0.9 x 0.9 metres centered at origin
"""
import time
import math
import os
import pyqtgraph.multiprocess as mp
from pyqtgraph.Qt import QtWidgets, QtCore
import sys

# A QApplication must exist in the main process before spawning a QtProcess
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

# Spawn child process with a Qt event loop
proc = mp.QtProcess()

# Import modules in the child process
rpg = proc._import('pyqtgraph')
np_remote = proc._import('numpy')
cv2_remote = proc._import('cv2')

# Create plot window in child
win = rpg.plot(title='Remote Background Image Test')
win.setAspectLocked(True)
win.setLabel('left', 'Y (m)')
win.setLabel('bottom', 'X (m)')

# Set view range to see the 0.9x0.9m image with some margin
margin = 0.15
win.setXRange(-0.45 - margin, 0.45 + margin)
win.setYRange(-0.45 - margin, 0.45 + margin)

# ---------------------------------------------------------------------------
# Background image setup
# ---------------------------------------------------------------------------
# Strategy: load the image path as a string (picklable), then have the child
# load it with its own cv2 import. ImageItem expects RGB data.

image_path = os.path.abspath('resources/gridboard_square.png')
print(f"Loading image: {image_path}")

# Load in child process via remote cv2
img_bgr = cv2_remote.imread(image_path)
img_rgb = cv2_remote.cvtColor(img_bgr, cv2_remote.COLOR_BGR2RGB)

# Create ImageItem in child — note: pyqtgraph expects (height, width, 3) shape
img_item = rpg.ImageItem(img_rgb)

# Scale the image to 0.9x0.9 metres, centered at origin
# setRect(x, y, w, h) where (x, y) is top-left corner in plot coordinates
# PyQtGraph image Y goes top-to-bottom, so y=-0.45 is correct for centering
size = 0.9
img_item.setRect(rpg.QtCore.QRectF(-size/2, -size/2, size, size))

# Add to the plot view (z-order: images default to background)
win.addItem(img_item)

# Add a red border rectangle to verify scaling
border = win.plot(
    x=[-size/2, size/2, size/2, -size/2, -size/2],
    y=[-size/2, -size/2, size/2, size/2, -size/2],
    pen=rpg.mkPen('r', width=2)
)

# Add a moving test curve to show plot is still live
curve = win.plot(pen='y')

print(f"Image displayed: {size}x{size}m centered at origin")
print("Red border shows exact image bounds. Yellow curve proves plot is live.")
print("Press Ctrl+C to quit.")
print()

t = 0.0
try:
    while True:
        t0 = time.monotonic()

        # Animated test curve
        xs = [i * 0.02 - 0.5 for i in range(50)]
        ys = [0.3 * math.sin(x * 10 + t) for x in xs]
        curve.setData(x=xs, y=ys, _callSync='off')

        elapsed = (time.monotonic() - t0) * 1000
        print(f"[main] {elapsed:.2f} ms")

        t += 0.1
        app.processEvents()
        time.sleep(0.02)

except KeyboardInterrupt:
    pass
finally:
    proc.close()
    print("Done.")
