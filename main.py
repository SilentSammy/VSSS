from mecanum_client import MecanumBLEClient, get_manual_override
from car_controller import load_cars
from combined_input import rising_edge
from cam_config import global_cam
from game_det import game_detector, global_plotter
from backg_poller import BackgroundPoller
import cv2
import time

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

def tracked_manual_mode():
    global_plotter.show()
    
    t_capture = time.monotonic()
    frame = global_cam.get_frame()
    t_capture = (time.monotonic() - t_capture) * 1000
    
    if frame is not None:
        # Poll for last detection result (non-blocking) and queue new frame
        t_detect = time.monotonic()
        game_state = detect_poller.poll(lambda: game_detector.detect(frame))
        t_detect = (time.monotonic() - t_detect) * 1000
        
        # Update plotter with last result (may be None on first frame)
        if game_state is not None:
            t_plotter = time.monotonic()
            game_state.balls = []  # Hide balls for now since detection is unreliable
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

# --- Main loop ---
MODES = [manual_mode, rotisserie_mode, tracked_manual_mode, camera_mode]
mode = 0
print(f"Mode: {MODES[mode].__name__} — press 1-{len(MODES)} to switch, ESC to quit")

try:
    while True:
        _t0 = time.monotonic()
        _tick_timers()
        if rising_edge('Key.esc'):
            break
        for i in range(len(MODES)):
            if rising_edge(str(i + 1)) and i != mode:
                mode = i
                print(f"Mode: {MODES[mode].__name__}")
        MODES[mode]()
        time.sleep(0.02)
        print(f"[loop] {(time.monotonic() - _t0)*1000:.1f} ms")
except KeyboardInterrupt:
    pass
finally:
    client.stop()
    client.disconnect()
