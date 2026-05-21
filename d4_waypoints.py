import cv2
import time
import numpy as np
from mecanum_client import MecanumBLEClient, get_manual_override
from cam_config import global_cam
from game_det import game_detector
from car_controller import CarController
from zmq_comms import PlotPublisher

DEVICE_NAME = 'Eq4'
ARUCO_ID    = 14

client     = MecanumBLEClient(device_name=DEVICE_NAME)
controller = CarController(
    kp_dist=5.0, ki_dist=0.0, kd_dist=0.2, max_speed=0.4,
    kp_heading=0.8, ki_heading=0.0, kd_heading=0.1, max_w=0.7,
)
client.connect()

waypoint = None

with PlotPublisher() as plotter:
    try:
        while True:
            clicks = plotter.get_clicks()
            if clicks:
                waypoint = clicks[-1]  # use the most recent click
                controller.reset()
                print(f"→ Navigating to ({waypoint[0]:.3f}, {waypoint[1]:.3f})")

            frame = global_cam.get_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            game_state = game_detector.detect(frame, include_balls=True)
            if game_state is None:
                time.sleep(0.02)
                continue

            player = next((p for p in game_state.players if p.id == ARUCO_ID), None)

            cmd = {'x': 0.0, 'y': 0.0, 'w': 0.0}

            if player is not None and waypoint is not None:
                tx, ty = waypoint
                cmd = controller.go_to(
                        player.x, player.y, player.angle,  # current pose
                        tx=tx,              # target x
                        ty=ty,              # target y
                        ttheta=np.pi / 2,   # target heading
                    )

            plotter.update(game_state, waypoint=waypoint)
            client.set_velocity(get_manual_override(cmd))

            cv2.imshow("Camera", frame)
            cv2.waitKey(1)  # allow OpenCV to process events

    except KeyboardInterrupt:
        pass
    finally:
        plotter.update(game_state, waypoint=None)
        client.stop()
        client.disconnect()
