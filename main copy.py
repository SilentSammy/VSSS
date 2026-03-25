import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rc'))
from mecanum_client import MecanumBLEClient, get_manual_override
import numpy as np
from collections import deque
import cv2
import time
from cam_config import global_cam
from car_controller import CarController, load_cars, PurePursuit, Path
from game_det import game_detector, global_plotter, PathOverlay
from combined_input import rising_edge

client = None

car = load_cars()[0]
car.controller = CarController()

def get_path():
    def heart(t, scale=0.6/32):
        x = 16 * np.sin(t)**3
        y = 13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)
        return x * scale, y * scale

    heart_path = Path.from_fn(heart, n=2000).resample(spacing=0.003).roll_to_closest(0.3, 0)
    inf_path   = Path.from_fn(lambda t: (
        0.284 * np.cos(t) / (1 + np.sin(t)**2),
        0.284 * np.sin(t) * np.cos(t) / (1 + np.sin(t)**2)
    ), n=1000).resample(spacing=0.003).reverse()

    # --- Pick a path (uncomment one) ---
    # return heart_path + inf_path
    # return heart_path
    # return inf_path
    return Path.circle(r=0.25).resample(spacing=0.005)
    # return Path.from_fn(lambda t: heart(t, 0.3/32), n=2000).resample(spacing=0.003)

path = get_path()

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

# ------------------------------------------------------------------ #
def manual_mode():
    client.set_velocity(get_manual_override({'x': 0, 'y': 0, 'w': 0}))
    time.sleep(0.02)

def waypoint_mode(threshold=0.03):
    waypoint_mode.waypoints = waypoint_mode.waypoints if hasattr(waypoint_mode, 'waypoints') else deque()
    waypoint_mode.current   = waypoint_mode.current   if hasattr(waypoint_mode, 'current')   else None
    waypoint_mode.overlay   = waypoint_mode.overlay   if hasattr(waypoint_mode, 'overlay')   else PathOverlay(global_plotter)

    frame = global_cam.get_frame()
    if frame is None:
        return
    game_state = game_detector.detect(frame)
    player = game_state.get_player(car.aruco_id)
    auto_cmd = {'x': 0, 'y': 0, 'w': 0}

    if not client.is_connected:
        car.controller.reset()
    elif player is not None:
        if waypoint_mode.current is None and waypoint_mode.waypoints:
            waypoint_mode.current = waypoint_mode.waypoints.popleft()
            car.controller.reset()
        if waypoint_mode.current is not None:
            tx, ty = waypoint_mode.current
            if np.hypot(tx - player.x, ty - player.y) < threshold:
                waypoint_mode.current = None
                car.controller.reset()
                tx, ty = None, None
        else:
            tx, ty = None, None
        auto_cmd = car.controller.go_to(
            player.x, player.y, player.angle,
            tx=tx, ty=ty, ttheta=np.radians(90)
        )

    points = ([waypoint_mode.current] if waypoint_mode.current else []) + list(waypoint_mode.waypoints)
    waypoint_mode.overlay.update(points, (player.x, player.y) if player is not None else None)
    global_plotter.update(game_state)
    client.set_velocity(get_manual_override(auto_cmd))
    cv2.pollKey()  # pump GUI events for Camera window
    cv2.imshow("Camera", frame)
    cv2.setWindowProperty("Camera", cv2.WND_PROP_TOPMOST, 1)

def path_mode(lookahead=0.05):
    path_mode.pursuit  = path_mode.pursuit  if hasattr(path_mode, 'pursuit')  else PurePursuit(path, lookahead=lookahead)
    path_mode.overlay  = path_mode.overlay  if hasattr(path_mode, 'overlay')  else PathOverlay(global_plotter, path_ms=2)
    path_mode.start_pt = path_mode.start_pt if hasattr(path_mode, 'start_pt') else tuple(path.pts[0])

    frame = global_cam.get_frame()
    if frame is None:
        return
    game_state = game_detector.detect(frame)
    player = game_state.get_player(car.aruco_id)
    auto_cmd = {'x': 0, 'y': 0, 'w': 0}

    if not client.is_connected:
        car.controller.reset()
        path_mode.overlay.update(path_mode.pursuit.ordered_points(), start_point=path_mode.start_pt)
    elif player is not None:
        tx, ty = path_mode.pursuit.get_target(player.x, player.y)
        auto_cmd = car.controller.go_to(
            player.x, player.y, player.angle,
            tx=tx, ty=ty, ttheta=np.radians(90),
            path_idx=path_mode.pursuit._idx
        )
        path_mode.overlay.update([(tx, ty)] + path_mode.pursuit.ordered_points(),
                                 player_pos=(player.x, player.y),
                                 start_point=path_mode.start_pt)
    else:
        path_mode.overlay.update(path_mode.pursuit.ordered_points(), start_point=path_mode.start_pt)

    global_plotter.update(game_state)
    client.set_velocity(get_manual_override(auto_cmd))
    cv2.pollKey()  # pump GUI events for Camera window
    cv2.imshow("Camera", frame)
    cv2.setWindowProperty("Camera", cv2.WND_PROP_TOPMOST, 1)
# ------------------------------------------------------------------ #

MODES = {
    'Manual':   manual_mode,
    'Waypoint': waypoint_mode,
    'Path':     path_mode,
}

def main():
    global_plotter.start()
    global_plotter.on_click = lambda x, y: waypoint_mode.waypoints.append((x, y))

    start_client(car.name)

    mode_names = list(MODES.keys())
    mode_funcs = list(MODES.values())
    mode = 0
    print(f"Mode: {mode_names[mode]} — press 1/2/3 to switch, ESC to quit")

    try:
        while True:
            if rising_edge('Key.esc'):
                break
            for i, key in enumerate(('1', '2', '3')):
                if rising_edge(key) and i != mode:
                    mode = i
                    car.controller.reset()
                    print(f"Mode: {mode_names[mode]}")

            mode_funcs[mode]()

    except KeyboardInterrupt:
        pass
    finally:
        if client is not None:
            client.set_velocity({'x': 0, 'y': 0, 'w': 0})
        stop_client()
        global_plotter.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
