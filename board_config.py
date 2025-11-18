"""
Centralized configuration for ArUco GridBoard.
Import the `board` object from this module to ensure consistent board parameters across the project.
"""

import cv2

# Load ArUco dictionary
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

# Board dimensions (for reference)
BOARD_WIDTH = 0.60   # 60 cm
MARKER_LENGTH = 0.075  # 7.5 cm per marker
MARKER_SEPARATION = (BOARD_WIDTH - 3 * MARKER_LENGTH) / 2  # calculated to fill width evenly

# Grid dimensions
GRID_SIZE = (3, 4)  # 3 columns × 4 rows = 12 markers

# Create the Board object
# board = cv2.aruco.GridBoard(
#     size=GRID_SIZE,
#     markerLength=MARKER_LENGTH,
#     markerSeparation=MARKER_SEPARATION,
#     dictionary=aruco_dict
# )
board = cv2.aruco.GridBoard(
    size=(3, 4),
    markerLength=0.075,
    markerSeparation=(0.60 - 3 * 0.075) / 2,
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
)
board = cv2.aruco.CharucoBoard(
    size=(9, 24),
    squareLength=0.1,
    markerLength=0.08,
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
)

# Detector
# detector = cv2.aruco.ArucoDetector(aruco_dict)
detector = cv2.aruco.CharucoDetector(board)