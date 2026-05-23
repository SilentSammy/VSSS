import time

from cam_config import global_cam
from game_det import game_detector
from zmq_comms import PlotPublisher

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
