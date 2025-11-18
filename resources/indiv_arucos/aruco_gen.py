#!/usr/bin/env python3
"""
Generate 4×4 ArUco markers as 120×120-pixel PNG files.

Dependencies:
    pip install --no-cache-dir --force-reinstall opencv-contrib-python
"""

import os
import sys

import cv2

def main():
    # 1) Verify the aruco module is present
    if not hasattr(cv2, "aruco"):
        sys.exit("ERROR: ArUco module not found. Install opencv-contrib-python, not just opencv-python.")

    # 2) Load the 4×4 dictionary (50 possible markers: IDs 0–49)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # 3) Draw and save markers for IDs 0 through n-1 in the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for marker_id in range(9, 13):
        # generateImageMarker creates the marker image
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 120)
        filename = os.path.join(script_dir, f"aruco_{marker_id:02d}.png")
        cv2.imwrite(filename, marker_img)
        print(f"Saved {filename}")

if __name__ == "__main__":
    main()
