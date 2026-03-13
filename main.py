import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rc'))
from mecanum_client import MecanumBLEClient, get_manual_override
import numpy as np
from cam_config import global_cam
from car_controller import CarController, load_cars, PurePursuit, Path

client = None

def start_client(device_name):
    global client
    if client is None:
        client = MecanumBLEClient(device_name=device_name)
        client.connect()

def stop_client():
    global client
    if client is not None:
        client.stop()
        client.disconnect()
        client = None

def waypoint_demo():
    from game_det import game_detector, global_plotter, PathOverlay
    from collections import deque
    import cv2

    game_detector.ball_detector = None  
    car = load_cars()[0]  # use first car from cars.json

    target_heading = 90.0  # degrees
    waypoint_threshold = 0.03  # 3cm
    waypoints = deque()
    current_waypoint = None

    global_plotter.on_click = lambda x, y: waypoints.append((x, y))
    global_plotter.start()
    path_overlay = PathOverlay(global_plotter)

    start_client(car.name)

    try:
        while True:
            if cv2.waitKey(1) & 0xFF == 27:
                break

            frame = global_cam.get_frame()
            if frame is None:
                continue

            game_state = game_detector.detect(frame)

            auto_cmd = {'x': 0, 'y': 0, 'w': 0}
            player = game_state.get_player(car.aruco_id)

            if player is not None:
                if current_waypoint is None and waypoints:
                    current_waypoint = waypoints.popleft()
                    car.controller.reset()

                if current_waypoint is not None:
                    tx, ty = current_waypoint
                    distance = np.hypot(tx - player.x, ty - player.y)
                    if distance < waypoint_threshold:
                        current_waypoint = None
                        car.controller.reset()
                        tx, ty = None, None
                else:
                    tx, ty = None, None

                auto_cmd = car.controller.go_to(
                    player.x, player.y, player.angle,
                    tx=tx, ty=ty, ttheta=np.radians(target_heading)
                )

            # Update path overlay
            points = ([current_waypoint] if current_waypoint else []) + list(waypoints)
            player_pos = (player.x, player.y) if player is not None else None
            path_overlay.update(points, player_pos)

            global_plotter.update(game_state)
            client.set_velocity(get_manual_override(auto_cmd))

    except KeyboardInterrupt:
        pass
    finally:
        if client is not None:
            client.set_velocity({'x': 0, 'y': 0, 'w': 0})
        stop_client()
        global_plotter.close()

def path_demo():
    from game_det import game_detector, global_plotter, PathOverlay
    import cv2

    game_detector.ball_detector = None
    car = load_cars()[0]

    def heart(t, scale=0.6/32):
        x = 16 * np.sin(t)**3
        y = 13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)
        return x * scale, y * scale

    # --- Pick a path (uncomment one) ---
    heart_path = Path.from_fn(lambda t: heart(t), n=2000).resample(spacing=0.003).roll_to_closest(0.3, 0)
    inf_path   = Path.from_fn(lambda t: (                          # lemniscate (infinity)
        0.284 * np.cos(t) / (1 + np.sin(t)**2),
        0.284 * np.sin(t) * np.cos(t) / (1 + np.sin(t)**2)
    ), n=1000).resample(spacing=0.003).reverse()
    # path = heart_path + inf_path
    # path = heart_path                                              # heart only
    # path = inf_path                                               # lemniscate only
    # path = Path.circle(r=0.25).resample(spacing=0.005)           # circle
    # path = Path.from_fn(lambda t: heart(t, 0.3/32), n=2000).resample(spacing=0.003)  # small heart
    # ------------------------------------
    path = inf_path
    
    pursuit = PurePursuit(path, lookahead=0.05)
    path_start = tuple(path.pts[0])

    global_plotter.start()
    path_overlay = PathOverlay(global_plotter, path_ms=2)

    start_client(car.name)

    try:
        while True:
            if cv2.waitKey(1) & 0xFF == 27:
                break

            frame = global_cam.get_frame()
            if frame is None:
                continue

            game_state = game_detector.detect(frame)

            auto_cmd = {'x': 0, 'y': 0, 'w': 0}
            player = game_state.get_player(car.aruco_id)

            if player is not None:
                tx, ty = pursuit.get_target(player.x, player.y)
                auto_cmd = car.controller.go_to(
                    player.x, player.y, player.angle,
                    tx=tx, ty=ty, ttheta=np.radians(90)  # constant heading
                )
                path_overlay.update([(tx, ty)] + pursuit.ordered_points(),
                                    player_pos=(player.x, player.y),
                                    start_point=path_start)
            else:
                path_overlay.update(pursuit.ordered_points(), start_point=path_start)

            global_plotter.update(game_state)
            client.set_velocity(get_manual_override(auto_cmd))

            cv2.imshow("Camera", frame)
            cv2.setWindowProperty("Camera", cv2.WND_PROP_TOPMOST, 1)

    except KeyboardInterrupt:
        pass
    finally:
        if client is not None:
            client.set_velocity({'x': 0, 'y': 0, 'w': 0})
        stop_client()
        global_plotter.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    path_demo()
