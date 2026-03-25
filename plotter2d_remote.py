"""
Remote-process GamePlotter2D using PyQtGraph multiprocess.

The plot window lives in a child process with its own Qt event loop.
All data pushes are fire-and-forget (_callSync='off'), keeping the
caller's main loop near 0 ms.

Lifecycle mirrors the original matplotlib GamePlotter2D:
    start()  — spawn child process, create window
    show()   — make the window visible
    hide()   — hide the window (update() becomes a no-op)
    stop()   — terminate the child process
    pump()   — call each loop iteration to service the IPC pipe
    update() — push new GameState; non-blocking
"""

import math
import sys


class GamePlotter2D:
    """Real-time 2D game-state plotter running in a separate Qt process.

    Balls are shown as orange circles.
    Players are shown as blue circles with a white direction arrow.

    Args:
        board_config: Optional board_config object (uses get_board_dimensions()).
                      If None, board_width / board_height are used directly.
        board_width:  Field width in metres (ignored when board_config is given).
        board_height: Field height in metres (ignored when board_config is given).
        margin:       Extra space around the field in metres.
        title:        Window title.
        ball_color:   (R, G, B) 0-255 ball fill colour.
        ball_size:    Scatter symbol diameter in pixels.
        player_color: (R, G, B) 0-255 player fill colour.
        player_size:  Scatter symbol diameter in pixels.
        arrow_length: Length of the player direction arrow in metres.
    """

    def __init__(self, board_config=None, *,
                 board_width=0.75, board_height=0.65, margin=0.05,
                 title='VSS Game State',
                 ball_color=(255, 165, 0), ball_size=12,
                 player_color=(50, 100, 255), player_size=14,
                 arrow_length=0.06):

        if board_config is not None:
            self._bw, self._bh = board_config.get_board_dimensions()
        else:
            self._bw, self._bh = board_width, board_height

        self.margin = margin
        self.title = title
        self.ball_color = ball_color
        self.ball_size = ball_size
        self.player_color = player_color
        self.player_size = player_size
        self.arrow_length = arrow_length

        self._app = None
        self._proc = None
        self._win = None
        self._balls_item = None
        self._players_item = None
        self._arrows_item = None

        self._started = False
        self._visible = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_started(self):
        return self._started

    @property
    def is_visible(self):
        return self._started and self._visible

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Spawn the child Qt process and open the plot window."""
        if self._started:
            self.show()
            return

        from pyqtgraph.Qt import QtWidgets
        import pyqtgraph.multiprocess as mp

        # A QApplication must exist in the parent before spawning QtProcess
        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

        self._proc = mp.QtProcess()
        rpg = self._proc._import('pyqtgraph')

        # --- build the remote plot ---
        self._win = rpg.plot(title=self.title)
        self._win.setAspectLocked(True)
        self._win.setLabel('left', 'Y (m)')
        self._win.setLabel('bottom', 'X (m)')

        bw, bh, m = self._bw, self._bh, self.margin
        self._win.setXRange(-bw / 2 - m, bw / 2 + m)
        self._win.setYRange(-bh / 2 - m, bh / 2 + m)

        # Static field boundary
        border_xs = [-bw/2, bw/2,  bw/2, -bw/2, -bw/2]
        border_ys = [-bh/2, -bh/2,  bh/2,  bh/2, -bh/2]
        self._win.plot(x=border_xs, y=border_ys, pen=rpg.mkPen('r', width=2))

        # Dynamic items
        br, bg, bb = self.ball_color
        pr, pg_, pb = self.player_color

        self._balls_item = self._win.plot(
            x=[], y=[],
            pen=None,
            symbol='o',
            symbolSize=self.ball_size,
            symbolBrush=rpg.mkBrush(br, bg, bb, 255),
            symbolPen=rpg.mkPen(None),
        )
        self._players_item = self._win.plot(
            x=[], y=[],
            pen=None,
            symbol='o',
            symbolSize=self.player_size,
            symbolBrush=rpg.mkBrush(pr, pg_, pb, 220),
            symbolPen=rpg.mkPen(None),
        )
        # Direction arrows: NaN-separated line segments, one per player
        self._arrows_item = self._win.plot(
            x=[], y=[],
            pen=rpg.mkPen('w', width=2),
        )

        self._started = True
        self._visible = True

    def show(self):
        """Make the plot window visible."""
        if not self._started:
            self.start()
            return
        if not self._visible:
            self._win.show(_callSync='off')
            self._visible = True

    def hide(self):
        """Hide the plot window. update() is a no-op while hidden."""
        if self._started and self._visible:
            self._win.hide(_callSync='off')
            self._visible = False

    def stop(self):
        """Terminate the child process and reset all state."""
        if not self._started:
            return
        if self._proc is not None:
            try:
                self._proc.close()
            except Exception:
                pass
        self._app = None
        self._proc = None
        self._win = None
        self._balls_item = None
        self._players_item = None
        self._arrows_item = None
        self._started = False
        self._visible = False

    # ------------------------------------------------------------------
    # Per-loop calls
    # ------------------------------------------------------------------

    def pump(self):
        """Service the IPC pipe between parent and child processes.

        Must be called once per main-loop iteration while the plotter is active.
        Equivalent to app.processEvents() in the parent's Qt event loop.
        """
        if self._app is not None:
            self._app.processEvents()

    def update(self, game_state):
        """Push new game state to the remote plot. Non-blocking.

        Args:
            game_state: Object with .balls (list of objects with .x/.y) and
                        .players (list of objects with .x/.y/.angle).
        """
        if not self._started or not self._visible:
            return

        # Balls — plain list of floats, safe to pickle across process boundary
        bxs = [b.x for b in game_state.balls]
        bys = [b.y for b in game_state.balls]
        self._balls_item.setData(x=bxs, y=bys, _callSync='off')

        # Player positions
        pxs = [p.x for p in game_state.players]
        pys = [p.y for p in game_state.players]
        self._players_item.setData(x=pxs, y=pys, _callSync='off')

        # Direction arrows — NaN-separated segments, one per player
        L = self.arrow_length
        axs, ays = [], []
        for p in game_state.players:
            axs += [p.x, p.x + L * math.cos(p.angle), float('nan')]
            ays += [p.y, p.y + L * math.sin(p.angle), float('nan')]
        self._arrows_item.setData(x=axs, y=ays, _callSync='off')


# ---------------------------------------------------------------------------
# Standalone test with dummy data
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import time
    from dataclasses import dataclass

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

    plotter = GamePlotter2D(board_width=0.75, board_height=0.65, margin=0.08)
    plotter.start()

    t = 0.0
    print("Remote plotter running. Ctrl+C to quit.")
    print()
    try:
        while True:
            t0 = time.monotonic()

            # One ball on a figure-8 (Lissajous)
            balls = [DummyBall(
                x=0.28 * math.sin(t),
                y=0.18 * math.sin(2 * t),
            )]

            # Two players orbiting the centre at opposite sides
            players = [
                DummyPlayer(
                    id=0,
                    x=0.22 * math.cos(t),
                    y=0.22 * math.sin(t),
                    angle=t + math.pi / 2,      # tangent direction
                ),
                DummyPlayer(
                    id=1,
                    x=0.22 * math.cos(t + math.pi),
                    y=0.22 * math.sin(t + math.pi),
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
