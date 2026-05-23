import time
import cv2

from cam_config import global_cam
from game_det import GameDetector
from board_est import BoardEstimator
from board_config import global_board_config
from obj_det import BallDetector, ArucoDetector
from zmq_comms import PlotPublisher

game_detector = GameDetector(
    board_estimator=BoardEstimator(global_board_config, K=global_cam.K, D=global_cam.D, rotate_180=True),
    ball_detector=BallDetector(),
    ball_height=0.02,
    aruco_detector=ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)),
    player_height=0.1,
)

with PlotPublisher() as plotter:
    try:
        while True:
            frame = global_cam.get_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            game_state = game_detector.detect(frame, include_balls=True)
            if game_state is None:
                time.sleep(0.02)
                continue

            plotter.update(game_state)
            time.sleep(0.02)

    except KeyboardInterrupt:
        pass
