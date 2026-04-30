"""
main.py — game logic loop (dummy).

ZMQ topology:
  BIND PUB  tcp://*:5556   → publishes GameState to plotter
  CONNECT SUB tcp://localhost:5557  ← receives ClickEvents from plotter
"""

import time
import numpy as np
import zmq
from path_utils import load_svg

from game_state import GameState, BallState, PlayerState, ClickEvent, PlotUpdate

GAME_STATE_PORT = 5556
CLICK_PORT      = 5557

context = zmq.Context()

# Outbound: game state
pub = context.socket(zmq.PUB)
pub.bind(f"tcp://*:{GAME_STATE_PORT}")

# Inbound: click events from plotter
sub = context.socket(zmq.SUB)
sub.connect(f"tcp://localhost:{CLICK_PORT}")
sub.setsockopt_string(zmq.SUBSCRIBE, '')

print(f"[main] Publishing game state on port {GAME_STATE_PORT}")
print(f"[main] Listening for clicks on port {CLICK_PORT}")

t = 0.0
last_time = time.perf_counter()
waypoint = None   # (x, y) updated by clicks from plotter
svg_points = load_svg('test_path.svg').tolist()   # sent once on first frame; plotter keeps it sticky

try:
    while True:
        now = time.perf_counter()
        dt = now - last_time
        last_time = now

        # --- Non-blocking check for incoming clicks ---
        try:
            while True:
                msg = sub.recv_string(zmq.NOBLOCK)
                click = ClickEvent.from_json(msg)
                waypoint = (click.x, click.y)
                print(f"\n[main] Waypoint set to ({click.x:.3f}, {click.y:.3f})")
        except zmq.Again:
            pass

        # --- Dummy game state using real dt ---
        balls = [
            BallState(x= 0.20 * np.cos(t),       y= 0.15 * np.sin(t)),
            BallState(x=-0.15 * np.cos(t * 1.5), y= 0.20 * np.sin(t * 1.5)),
        ]
        players = [
            PlayerState(id=0, x= 0.30 * np.cos(t * 0.5), y= 0.20 * np.sin(t * 0.5), angle=t * 0.5),
            PlayerState(id=1, x=-0.25,                    y= 0.10 * np.sin(t),        angle=np.pi + t),
            PlayerState(id=2, x= 0.25,                    y=-0.15 * np.cos(t * 0.7), angle=t * 2),
        ]
        game_state = GameState(balls=balls, players=players, timestamp=time.time())
        plot_update = PlotUpdate(
            game_state=game_state,
            svg_points=svg_points,
            waypoint=waypoint,
        )

        pub.send_string(plot_update.to_json())

        # --- Advance simulation time by real elapsed time ---
        t += dt
        print(f"\r[main] Game loop: {1/dt:.1f} Hz   ", end='', flush=True)

except KeyboardInterrupt:
    print("\n[main] Stopped")
finally:
    pub.close()
    sub.close()
    context.term()
