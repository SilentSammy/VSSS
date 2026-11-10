# Support both `python -m detection.NAME` and direct script launches.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "detection"

import os

import numpy as np
import cv2
from . import cam_config
from . import board_config
from . import board_est
from . import plotter2d
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(kw_only=True)
class VisionSetup:
    camera: object
    board: board_config.BoardConfig
    detector: cv2.aruco.ArucoDetector
    marker_height_m: float
    marker_config_path: Path | str | None = None
    plot_margin_m: float = 0.2
    rotate_180: bool = False
    board_estimator: board_est.BoardEstimator = field(init=False, repr=False)
    _plotter: plotter2d.BoardPlotter2D | None = field(
        init=False, default=None, repr=False
    )

    def __post_init__(self):
        self.board_estimator = board_est.BoardEstimator(
            board_config=self.board,
            K=self.camera.K,
            D=self.camera.D,
            rotate_180=self.rotate_180,
        )

    @property
    def plotter(self):
        """Create and cache the 2D plotter only when it is first requested."""
        if self._plotter is None:
            self._plotter = plotter2d.BoardPlotter2D(
                self.board,
                self.detector.getDictionary(),
                margin=self.plot_margin_m,
            )
        return self._plotter

def _get_webcam_image():
    source = getattr(_get_webcam_image, "source", None)
    if source is None:
        source = _webcam.open_capture()
        _get_webcam_image.source = source
        _get_webcam_image.cap = source.cap
    return source.read()

_webcam = cam_config.Camera(
    K=np.array([[735.09668766, 0., 308.18011975], [0., 735.62248422, 242.58646203], [0., 0., 1.]], dtype=np.float32),
    D=np.array([0.15017654, -1.34648531, 0.00405315, -0.00410719, 2.41472656], dtype=np.float32),
    frame_getter=_get_webcam_image,
    device_candidates=(1, 0),
    capture_size_wh=(1920, 1080),
    capture_fps=30,
    capture_fourcc="MJPG",
    calibration_size_wh=(640, 480),
    # Measured on this camera: 1080p reveals more horizontal field of view.
    calibration_fit="expand",
)

board_letter = board_config.GridboardConfig(
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50),
    size=(3, 4),
    marker_length=0.025,
    board_width=0.15,      # board content width (fits Letter height)
    print_width=0.2159,    # Letter width is 8.5" = 21.59cm
    filename=str(Path(__file__).parent / "resources/board_letter")
)

board_90cm = board_config.GridboardConfig(
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50),
    size=(3, 4),
    marker_length=0.1,
    board_width=0.84,
    print_width=0.9,
    filename=str(Path(__file__).parent / "resources/board_90cm")
)

board_160cm = board_config.GridboardConfig(
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50),
    size=(3, 5),
    marker_length=0.1,
    board_width=1.5,
    print_width=1.6,
    filename=str(Path(__file__).parent / "resources/board_160cm")
)


_aruco_detector4 = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
_aruco_detector5 = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50))
_aruco_detector6 = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50))

VISION_SETUPS = {
    "setup_letter": VisionSetup(
        camera=_webcam,
        board=board_letter,
        detector=_aruco_detector4,
        marker_height_m=0.12,
        marker_config_path=Path(__file__).with_name("marker_config.json"),
    ),
    "setup_90cm": VisionSetup(
        camera=_webcam,
        board=board_90cm,
        detector=_aruco_detector4,
        marker_height_m=0.12,
        marker_config_path=Path(__file__).with_name("marker_config.json"),
    ),
    "setup_160cm": VisionSetup(
        camera=_webcam,
        board=board_160cm,
        detector=_aruco_detector4,
        marker_height_m=0.12,
        marker_config_path=Path(__file__).with_name("marker_config.json"),
    ),
}

DEFAULT_SETUP = VISION_SETUPS["setup_160cm"]


if __name__ == "__main__":
    setup = DEFAULT_SETUP
    if (input("Generate board? (y/n):")).lower() == "y":
        board = setup.board
        print(board.get_print_dimensions())  # Print dimensions including margins
        img = board.generate_image(filepath=board.image_path, width_px=2160*6)
        pdf_path = board.generate_vector_pdf(board.pdf_path)
        marker_paths = board.generate_marker_images(side_px=1200)
        print(f"Saved board image: {board.image_path}")
        print(f"Saved vector board: {pdf_path}")
        print(f"Saved {len(marker_paths)} Google Drawings marker PNGs in "
              f"{os.path.dirname(marker_paths[0])}")
    
    if (input("Generate individual markers? (y/n):")).lower() == "y":
        # Draw and save markers for IDs 0 through n-1 in the script's directory
        aruco_dict = setup.detector.getDictionary()
        script_dir = str(Path(__file__).parent / "resources/individual_markers")
        os.makedirs(script_dir, exist_ok=True)
        for marker_id in range(20):
            # generateImageMarker creates the marker image
            marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 1200)
            marker_img = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
            filename = os.path.join(script_dir, f"aruco_{marker_id:02d}.png")
            cv2.imwrite(filename, marker_img)
            print(f"Saved {filename}")
