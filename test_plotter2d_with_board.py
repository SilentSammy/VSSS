"""
Test plotter2d_remote.py with real board_config (including background image).

This proves the remote plotter can:
- Load board_config dimensions
- Display the background image scaled correctly
- Show the red border at print dimensions
- Update balls and players on top of the image
"""
import time
import math
from dataclasses import dataclass
from board_config import global_board_config
from plotter2d_remote import GamePlotter2D

@dataclass
class DummyBall:
    x: float
    y: float

@dataclass
class DummyPlayer:
    id: int
    x: float
    y: float
    angle: float

class DummyGameState:
    def __init__(self, balls, players):
        self.balls = balls
        self.players = players

# Create plotter with real board_config (includes image path, dimensions)
plotter = GamePlotter2D(global_board_config)
plotter.start()

t = 0.0
print("Remote plotter with board_config running. Ctrl+C to quit.")
print(f"Board dimensions: {global_board_config.get_board_dimensions()}")
print(f"Print dimensions: {global_board_config.get_print_dimensions()}")
print(f"Image path: {global_board_config.image_path}")
print()

try:
    while True:
        t0 = time.monotonic()

        # Scale motion to board size
        bw, bh = global_board_config.get_board_dimensions()
        
        # One ball on a figure-8
        balls = [DummyBall(
            x=(bw * 0.35) * math.sin(t),
            y=(bh * 0.28) * math.sin(2 * t),
        )]

        # Two players orbiting
        r = min(bw, bh) * 0.3
        players = [
            DummyPlayer(
                id=0,
                x=r * math.cos(t),
                y=r * math.sin(t),
                angle=t + math.pi / 2,
            ),
            DummyPlayer(
                id=1,
                x=r * math.cos(t + math.pi),
                y=r * math.sin(t + math.pi),
                angle=t + math.pi * 3 / 2,
            ),
        ]

        state = DummyGameState(balls, players)
        plotter.update(state)
        plotter.pump()

        elapsed = (time.monotonic() - t0) * 1000
        print(f"[loop] {elapsed:.2f} ms")

        t += 0.05
        time.sleep(0.02)

except KeyboardInterrupt:
    pass
finally:
    plotter.stop()
    print("Done.")
