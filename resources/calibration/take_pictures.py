import sys
import os
sys.path.extend([os.path.join(os.path.dirname(os.path.abspath(__file__)), *(['..'] * i)) for i in range(3)]) # add range-1 parent dirs to path

import cv2
from cam_config import global_cam
import time

last_time = 0
while True:
    now = time.time()
    img = global_cam.get_frame()

    cv2.imshow("Calibration", img)

    if now - last_time >= 1.0:
        # Save image every second
        filename = f"resources/calibration/capture_{int(now)}.jpg"
        cv2.imwrite(filename, img)
        print(f"Saved {filename}")
        last_time = now

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
