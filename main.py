from mecanum_client import MecanumBLEClient, get_manual_override
from car_controller import load_cars, PurePursuit
from combined_input import rising_edge, is_pressed
from cam_config import global_cam
from game_det import game_detector
from backg_poller import BackgroundPoller
from zmq_comms import PlotGameState, PlotBallState, PlotPlayerState, PlotUpdate, ClickEvent
from path_utils import load_svg
from collections import deque
import numpy as np
import zmq
import cv2
import time

GAME_STATE_PORT = 5556
CLICK_PORT      = 5557

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

# ZMQ sockets: publish game state to plotter process, subscribe to clicks
zmq_ctx = zmq.Context()
pub = zmq_ctx.socket(zmq.PUB)
pub.bind(f"tcp://*:{GAME_STATE_PORT}")
sub = zmq_ctx.socket(zmq.SUB)
sub.connect(f"tcp://localhost:{CLICK_PORT}")
sub.setsockopt_string(zmq.SUBSCRIBE, '')

# Background game detection
detect_poller = BackgroundPoller(max_workers=1)
_last_plot_gs = None  # last published PlotGameState, used for overlay-only updates

def _publish(game_state, **overlay_kwargs):
    """Publish a PlotUpdate, caching the game state for overlay-only sends."""
    global _last_plot_gs
    _last_plot_gs = game_state
    pub.send_string(PlotUpdate(game_state=game_state, **overlay_kwargs).to_json())

def _publish_overlay(**overlay_kwargs):
    """Send an overlay-only update using the last known game state (if any)."""
    if _last_plot_gs is not None:
        pub.send_string(PlotUpdate(game_state=_last_plot_gs, **overlay_kwargs).to_json())

# --- Modes ---
def manual_mode():
    client.set_velocity(get_manual_override({'x': 0, 'y': 0, 'w': 0}))

def rotisserie_mode():
    client.set_velocity(get_manual_override({'x': 0, 'y': 0, 'w': 0.2}))

# --- Waypoint mode state ---
_waypoints = deque()
_current_waypoint = None
WAYPOINT_THRESHOLD = 0.03  # metres

def waypoint_mode():
    global _current_waypoint

    # Drain click events from plotter (non-blocking)
    try:
        while True:
            msg = sub.recv_string(flags=zmq.NOBLOCK)
            event = ClickEvent.from_json(msg)
            _waypoints.append((event.x, event.y))
            print(f"✓ Waypoint added: ({event.x:.3f}, {event.y:.3f}) — queue: {len(_waypoints)}")
    except zmq.Again:
        pass

    frame = global_cam.get_frame()
    game_state = None
    if frame is not None:
        game_state = detect_poller.poll(lambda: game_detector.detect(frame, include_balls=False))

    auto_cmd = {'x': 0.0, 'y': 0.0, 'w': 0.0}
    wp_for_plot = None  # None = keep previous sticky waypoint

    if game_state is not None:
        player = next((p for p in game_state.players if p.id == car.aruco_id), None)

        if player is not None:
            if _current_waypoint is None and _waypoints:
                _current_waypoint = _waypoints.popleft()
                wp_for_plot = _current_waypoint
                car.controller.reset()
                print(f"→ Navigating to ({_current_waypoint[0]:.3f}, {_current_waypoint[1]:.3f})")

            if _current_waypoint is not None:
                tx, ty = _current_waypoint
                dist = np.hypot(tx - player.x, ty - player.y)
                if dist < WAYPOINT_THRESHOLD:
                    print("✓ Waypoint reached!")
                    _current_waypoint = None
                    wp_for_plot = ()  # clear marker on plotter
                    car.controller.reset()
                else:
                    auto_cmd = car.controller.go_to(
                        player.x, player.y, player.angle, tx=tx, ty=ty, ttheta=np.pi / 2
                    )

        plot_gs = PlotGameState(
            balls=[PlotBallState(x=b.x, y=b.y) for b in game_state.balls],
            players=[PlotPlayerState(id=p.id, x=p.x, y=p.y, angle=p.angle) for p in game_state.players],
            timestamp=game_state.timestamp,
        )
        _publish(plot_gs, waypoint=wp_for_plot)

    reset_timer('waypoint_active', 0.1, lambda: _publish_overlay(waypoint=()))
    client.set_velocity(get_manual_override(auto_cmd))

# --- Path mode state ---
# TODO: Add auto-loadinf from directory
PATH_SVGS = [ 'resources/paths/heart.svg', ]
_path_idx = 0
_pursuit = None
_path_pts_list = None

def _load_path(idx):
    global _pursuit, _path_pts_list, _path_idx
    _path_idx = idx
    pts = load_svg(PATH_SVGS[idx])
    _pursuit = PurePursuit(pts, lookahead=0.1, loop=True)
    _path_pts_list = pts.tolist()
    car.controller.reset()
    name = PATH_SVGS[idx].split('/')[-1]
    print(f"[path] {idx+1}/{len(PATH_SVGS)}: {name} ({len(pts)} pts)")

# TODO: Add hotkey to temporarily turn off path following and do manual control (e.g. for repositioning)
def path_mode():
    # Switch path with keys 1-n (no Alt)
    for i in range(len(PATH_SVGS)):
        if rising_edge(str(i + 1)) and i != _path_idx:
            _load_path(i)
            break

    if _pursuit is None:
        _load_path(0)

    frame = global_cam.get_frame()
    game_state = None
    if frame is not None:
        game_state = detect_poller.poll(lambda: game_detector.detect(frame, include_balls=False))

    auto_cmd = {'x': 0.0, 'y': 0.0, 'w': 0.0}

    if game_state is not None:
        player = next((p for p in game_state.players if p.id == car.aruco_id), None)

        if player is not None:
            tx, ty = _pursuit.get_target(player.x, player.y)
            auto_cmd = car.controller.go_to(
                player.x, player.y, player.angle, tx=tx, ty=ty, ttheta=np.pi / 2
            )

        plot_gs = PlotGameState(
            balls=[PlotBallState(x=b.x, y=b.y) for b in game_state.balls],
            players=[PlotPlayerState(id=p.id, x=p.x, y=p.y, angle=p.angle) for p in game_state.players],
            timestamp=game_state.timestamp,
        )
        wp = (tx, ty) if player is not None else None
        _publish(plot_gs, svg_points=_path_pts_list, waypoint=wp)

    reset_timer('path_active', 0.1, lambda: _publish_overlay(svg_points=[], waypoint=()))
    client.set_velocity(get_manual_override(auto_cmd))

def camera_mode():
    frame = global_cam.get_frame()
    if frame is not None:
        cv2.imshow("Camera", frame)
        cv2.setWindowProperty("Camera", cv2.WND_PROP_TOPMOST, 1)
    cv2.pollKey()
    client.set_velocity(get_manual_override({'x': 0, 'y': 0, 'w': 0}))
    reset_timer('camera', 0.3, lambda: cv2.destroyWindow("Camera"))

def plot_mode():
    t_capture = time.monotonic()
    frame = global_cam.get_frame()
    t_capture = (time.monotonic() - t_capture) * 1000

    if frame is not None:
        t_detect = time.monotonic()
        game_state = detect_poller.poll(lambda: game_detector.detect(frame, include_balls=False))
        t_detect = (time.monotonic() - t_detect) * 1000

        if game_state is not None:
            t_plotter = time.monotonic()
            plot_gs = PlotGameState(
                balls=[PlotBallState(x=b.x, y=b.y) for b in game_state.balls],
                players=[PlotPlayerState(id=p.id, x=p.x, y=p.y, angle=p.angle) for p in game_state.players],
                timestamp=game_state.timestamp,
            )
            _publish(plot_gs)
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
    reset_timer('camera', 0.3, lambda: cv2.destroyWindow("Camera"))

# --- Main loop ---
MODES = [manual_mode, rotisserie_mode, plot_mode, camera_mode, waypoint_mode, path_mode]

mode = 0
print(f"Mode: {MODES[mode].__name__} — press 1-{len(MODES)} to switch, ESC to quit")

try:
    while True:
        _t0 = time.monotonic()
        _tick_timers()
        if rising_edge('Key.esc'):
            break
        if is_pressed('Key.alt_l'):
            for i in range(len(MODES)):
                if rising_edge(str(i + 1)) and i != mode:
                    mode = i
                    print(f"Mode: {MODES[mode].__name__}")
        MODES[mode]()
        time.sleep(0.02)
        # print(f"[loop] {(time.monotonic() - _t0)*1000:.1f} ms")
except KeyboardInterrupt:
    pass
finally:
    client.stop()
    client.disconnect()
    pub.close()
    sub.close()
    zmq_ctx.term()
    cv2.destroyAllWindows()
