from mecanum_client import MecanumBLEClient, get_manual_override
from car_controller import load_cars
from combined_input import rising_edge, is_pressed
from cam_config import global_cam
from game_det import game_detector, global_plotter
from backg_poller import BackgroundPoller
import cv2
import time
import numpy as np
from collections import deque

# --- Timer system ---
_timers = {}  # key -> (deadline, callback)

def reset_timer(key, delay, fn):
    _timers[key] = (time.monotonic() + delay, fn)

def _tick_timers():
    now = time.monotonic()
    fired = [k for k, (deadline, _) in _timers.items() if now >= deadline]
    for k in fired:
        _, fn = _timers.pop(k)
        fn()

# --- Setup ---
car = load_cars()[0]
client = MecanumBLEClient(device_name=car.name)
client.connect()

global_plotter.start()  # Window is created, ready for instant show()
global_plotter.hide()   # Ensure it's hidden after starting

# Background game detection
detect_poller = BackgroundPoller(max_workers=1)

# Waypoint tracking
waypoints = deque()
current_waypoint = None
waypoint_threshold = 0.03  # 3cm
target_heading = np.radians(90.0)  # Face forward

# Waypoint visualization overlays (created after plotter.start())
path_overlay_items = None

def _cleanup_waypoint_mode():
    """Clear waypoints and path overlays."""
    global current_waypoint
    waypoints.clear()
    current_waypoint = None
    car.controller.reset()
    if path_overlay_items is not None:
        for item in path_overlay_items.values():
            item.setData(x=[], y=[], _callSync='off')

def _setup_path_overlays():
    """Create overlay items for waypoint visualization."""
    global path_overlay_items
    if path_overlay_items is None and global_plotter.is_started:
        rpg = global_plotter.rpg
        path_overlay_items = {
            'pursuit_line': global_plotter.add_overlay(pen=rpg.mkPen('y', width=2)),
            'target_dot': global_plotter.add_overlay(
                pen=None, symbol='o', symbolSize=20,
                symbolBrush=rpg.mkBrush(255, 255, 0, 200),
                symbolPen=rpg.mkPen('y', width=2)
            ),
            'path_line': global_plotter.add_overlay(
                pen=rpg.mkPen('c', width=2, style=rpg.QtCore.Qt.DashLine)
            ),
            'path_dots': global_plotter.add_overlay(
                pen=None, symbol='o', symbolSize=12,
                symbolBrush=rpg.mkBrush(0, 255, 255, 150)
            ),
        }

# --- Modes ---
def manual_mode():
    client.set_velocity(get_manual_override({'x': 0, 'y': 0, 'w': 0}))

def rotisserie_mode():
    client.set_velocity(get_manual_override({'x': 0, 'y': 0, 'w': 0.2}))

def camera_mode():
    frame = global_cam.get_frame()
    if frame is not None:
        cv2.imshow("Camera", frame)
        cv2.setWindowProperty("Camera", cv2.WND_PROP_TOPMOST, 1)
    cv2.pollKey()
    client.set_velocity(get_manual_override({'x': 0, 'y': 0, 'w': 0}))
    reset_timer('camera', 0.3, lambda: cv2.destroyWindow("Camera"))

def plot_mode():
    global_plotter.show()
    
    t_capture = time.monotonic()
    frame = global_cam.get_frame()
    t_capture = (time.monotonic() - t_capture) * 1000
    
    if frame is not None:
        # Poll for last detection result (non-blocking) and queue new frame
        t_detect = time.monotonic()
        game_state = detect_poller.poll(lambda: game_detector.detect(frame, include_balls=False))
        t_detect = (time.monotonic() - t_detect) * 1000
        
        # Update plotter with last result (may be None on first frame)
        if game_state is not None:
            t_plotter = time.monotonic()
            global_plotter.update(game_state)
            t_plotter = (time.monotonic() - t_plotter) * 1000
        else:
            t_plotter = 0.0
        
        t_cv2 = time.monotonic()
        cv2.imshow("Camera", frame)
        cv2.setWindowProperty("Camera", cv2.WND_PROP_TOPMOST, 1)
        t_cv2 = (time.monotonic() - t_cv2) * 1000
        
        print(f"  [capture] {t_capture:.1f}ms [detect] {t_detect:.1f}ms [plotter] {t_plotter:.1f}ms [cv2] {t_cv2:.1f}ms")
    
    cv2.pollKey()
    client.set_velocity(get_manual_override({'x': 0, 'y': 0, 'w': 0}))
    reset_timer('plotter', 0.3, global_plotter.hide)
    reset_timer('camera', 0.3, lambda: cv2.destroyWindow("Camera"))

def waypoint_mode():
    """Autonomous waypoint following. Click on plotter to add waypoints."""
    global current_waypoint
    
    global_plotter.show()
    _setup_path_overlays()
    
    # Check for PID parameter updates
    car.controller.check_tune_file()
    
    auto_cmd = {'x': 0, 'y': 0, 'w': 0}  # Default command
    
    frame = global_cam.get_frame()
    if frame is not None:
        # Get game state from background poller
        game_state = detect_poller.poll(lambda: game_detector.detect(frame, include_balls=False))
        
        if game_state is not None:
            # Check for new waypoint clicks
            click = global_plotter.check_click()
            if click is not None:
                waypoints.append(click)
                print(f"  [WAYPOINT] Added ({click[0]:.3f}, {click[1]:.3f}) — total: {len(waypoints) + (1 if current_waypoint else 0)}")
            
            # Get our player
            player = game_state.get_player(car.aruco_id)
            
            if player is not None:
                # Move to next waypoint if we reached current one
                if current_waypoint is None and waypoints:
                    current_waypoint = waypoints.popleft()
                    car.controller.reset()
                    print(f"  [TARGET] Moving to ({current_waypoint[0]:.3f}, {current_waypoint[1]:.3f})")
                
                if current_waypoint is not None:
                    tx, ty = current_waypoint
                    distance = np.hypot(tx - player.x, ty - player.y)
                    if distance < waypoint_threshold:
                        print(f"  [REACHED] Waypoint at ({tx:.3f}, {ty:.3f})")
                        current_waypoint = None
                        car.controller.reset()
                        tx, ty = None, None
                else:
                    tx, ty = None, None
                
                # Generate control command
                auto_cmd = car.controller.go_to(
                    player.x, player.y, player.angle,
                    tx=tx, ty=ty, ttheta=target_heading
                )
                
                # Update path visualization
                points = ([current_waypoint] if current_waypoint else []) + list(waypoints)
                if points:
                    # Target dot
                    path_overlay_items['target_dot'].setData(
                        x=[points[0][0]], y=[points[0][1]], _callSync='off'
                    )
                    # Pursuit line from player to target
                    path_overlay_items['pursuit_line'].setData(
                        x=[player.x, points[0][0]], y=[player.y, points[0][1]], _callSync='off'
                    )
                    # Path line through all points
                    if len(points) > 1:
                        path_xs, path_ys = zip(*points)
                        path_overlay_items['path_line'].setData(x=path_xs, y=path_ys, _callSync='off')
                        # Dots on remaining waypoints
                        rest_xs, rest_ys = zip(*points[1:])
                        path_overlay_items['path_dots'].setData(x=rest_xs, y=rest_ys, _callSync='off')
                    else:
                        path_overlay_items['path_line'].setData(x=[], y=[], _callSync='off')
                        path_overlay_items['path_dots'].setData(x=[], y=[], _callSync='off')
                else:
                    # No waypoints, hide all overlays
                    for item in path_overlay_items.values():
                        item.setData(x=[], y=[], _callSync='off')
            
            global_plotter.update(game_state)
        
        cv2.imshow("Camera", frame)
        cv2.setWindowProperty("Camera", cv2.WND_PROP_TOPMOST, 1)
    
    cv2.pollKey()
    client.set_velocity(get_manual_override(auto_cmd))
    reset_timer('plotter', 0.3, global_plotter.hide)
    reset_timer('camera', 0.3, lambda: cv2.destroyWindow("Camera"))

# --- Main loop ---
MODES = [manual_mode, rotisserie_mode, plot_mode, camera_mode, waypoint_mode]
mode = 0
print(f"Mode: {MODES[mode].__name__} — press 1-{len(MODES)} to switch, ESC to quit")
print("Waypoint mode (5): Click on plotter to add waypoints")

try:
    while True:
        _t0 = time.monotonic()
        _tick_timers()
        if rising_edge('Key.esc'):
            break
        if is_pressed('Key.alt_l'):
            for i in range(len(MODES)):
                if rising_edge(str(i + 1)) and i != mode:
                    old_mode = mode
                    mode = i
                    print(f"Mode: {MODES[mode].__name__}")
                    
                    # Trigger waypoint cleanup when leaving waypoint mode
                    if old_mode == 4 and mode != 4:
                        reset_timer('waypoint_cleanup', 0, _cleanup_waypoint_mode)
        MODES[mode]()
        time.sleep(0.02)
        print(f"[loop] {(time.monotonic() - _t0)*1000:.1f} ms")
except KeyboardInterrupt:
    pass
finally:
    client.stop()
    client.disconnect()
    global_plotter.stop()
    cv2.destroyAllWindows()
