"""
Remote-process GamePlotter2D using PyQtGraph multiprocess.

The plot window lives in a child process with its own Qt event loop.
All data pushes are fire-and-forget (_callSync='off'), keeping the
caller's main loop near 0 ms.

A background thread handles IPC communication, so pump() calls never block.
The thread processes the most recent update and drops old ones if busy.

Lifecycle mirrors the original matplotlib GamePlotter2D:
    start()       — spawn child process, create window, start worker thread
    show()        — make the window visible
    hide()        — hide the window (update() becomes a no-op)
    stop()        — terminate the child process and worker thread
    update()      — push new GameState; never blocks (thread-safe)
    check_click() — poll for click events; returns (x, y) or None
    add_overlay() — create custom overlay items for external drawing

External overlays:
    Users can add custom drawings by calling add_overlay() which returns
    an ObjectProxy to a PlotDataItem. Update it each frame with:
        overlay.setData(x=xs, y=ys, _callSync='off')
"""

import math
import sys
import os
import tempfile
import threading
import queue


# ---------------------------------------------------------------------------
# Click detection helper (runs in child process)
# ---------------------------------------------------------------------------

_click_file = None
_vb = None


def _click_store_setup(win, click_file):
    """Register a click handler on the plot window (called in child process).
    
    The parent polls the file's mtime each loop; when it changes, it reads
    the latest click coordinates. No queues, no Manager, no IPC complexity.

    Args:
        win:        PlotWidget (real object, lives in child process)
        click_file: Absolute path string to write click coords to
    """
    global _click_file, _vb
    _click_file = click_file
    _vb = win.getViewBox()
    win.scene().sigMouseClicked.connect(_click_store_on_click)


def _click_store_on_click(event):
    """Handle mouse click event in child process."""
    if _click_file is None or _vb is None:
        return
    pos = _vb.mapSceneToView(event.scenePos())
    try:
        with open(_click_file, 'w') as f:
            f.write(f'{float(pos.x())},{float(pos.y())}\n')
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main plotter class
# ---------------------------------------------------------------------------

class GamePlotter2D:
    """Real-time 2D game-state plotter running in a separate Qt process.

    Balls are shown as orange circles.
    Players are shown as blue circles with a cyan direction arrow and yellow ID number.

    Args:
        board_config: BoardConfig object providing:
                      - get_board_dimensions() for field size
                      - get_print_dimensions() for image scaling
                      - image_path for background image
        margin:       Extra space around the field in metres.
        title:        Window title.
        ball_color:   (R, G, B) 0-255 ball fill colour.
        ball_size:    Scatter symbol diameter in pixels.
        player_color: (R, G, B) 0-255 player fill colour.
        player_size:  Scatter symbol diameter in pixels.
        arrow_length: Length of the player direction arrow in metres.
    """

    def __init__(self, board_config, *,
                 margin=0.05,
                 title='VSS Game State',
                 ball_color=(255, 165, 0), ball_size=15,
                 player_color=(50, 100, 255), player_size=30,
                 arrow_length=0.06):

        self.board_config = board_config
        self._bw, self._bh = board_config.get_board_dimensions()
        self._pw, self._ph = board_config.get_print_dimensions()
        self._image_path = board_config.image_path if hasattr(board_config, 'image_path') else None

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
        self._player_text_items = []  # Dynamic text labels for player IDs

        self._started = False
        self._visible = False

        # Click detection
        self._click_file = os.path.join(tempfile.gettempdir(), 'vss_click.txt')
        self._last_click_mtime = 0.0

        # Worker thread for non-blocking IPC
        self._worker_thread = None
        self._update_queue = None
        self._stop_worker = False
        self._worker_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_started(self):
        return self._started

    @property
    def is_visible(self):
        return self._started and self._visible

    @property
    def rpg(self):
        """Access to the remote pyqtgraph module for creating pens, brushes, etc.

        Use this to create styling objects for custom overlays:
            pen = plotter.rpg.mkPen('c', width=2, style=plotter.rpg.QtCore.Qt.DashLine)
            overlay = plotter.add_overlay(pen=pen)

        Returns None if the plotter has not been started yet.
        """
        return self._proc._import('pyqtgraph') if self._started else None

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

        # --- Background image (if present) ---
        if self._image_path is not None:
            import os
            if os.path.exists(self._image_path):
                cv2_remote = self._proc._import('cv2')
                
                # Load image in child process
                img_bgr = cv2_remote.imread(self._image_path)
                img_rgb = cv2_remote.cvtColor(img_bgr, cv2_remote.COLOR_BGR2RGB)
                
                # Create ImageItem and scale to physical print dimensions
                img_item = rpg.ImageItem(img_rgb)
                pw, ph = self._pw, self._ph
                img_item.setRect(rpg.QtCore.QRectF(-pw/2, -ph/2, pw, ph))
                self._win.addItem(img_item)
                
                # Red border showing print area (matches original GamePlotter2D)
                self._win.plot(
                    x=[-pw/2, pw/2,  pw/2, -pw/2, -pw/2],
                    y=[-ph/2, -ph/2,  ph/2,  ph/2, -ph/2],
                    pen=rpg.mkPen('r', width=2)
                )
        else:
            # No image: draw field boundary
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
            pen=rpg.mkPen('c', width=2),
        )

        # --- Click detection setup ---
        plotter_remote = self._proc._import('plotter2d_remote')
        plotter_remote._click_store_setup(self._win, self._click_file)

        # --- Start worker thread for non-blocking IPC ---
        self._update_queue = queue.Queue(maxsize=1)  # Only holds 1 item (latest)
        self._stop_worker = False
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

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
        
        # Stop worker thread
        self._stop_worker = True
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        
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
        self._player_text_items = []
        self._update_queue = None
        self._worker_thread = None
        self._started = False
        self._visible = False

    def _worker_loop(self):
        """Worker thread: processes update queue and handles IPC non-blocking."""
        while not self._stop_worker:
            try:
                # Wait for an update with timeout so we can check stop flag
                update_data = self._update_queue.get(timeout=0.05)
                
                # Process the update (IPC happens here, in worker thread)
                self._process_update(update_data)
                
                # Service the IPC pipe
                if self._app is not None:
                    self._app.processEvents()
                    
            except queue.Empty:
                # No update available, service pipe anyway
                if self._app is not None:
                    self._app.processEvents()
            except Exception as e:
                # Log but don't crash worker thread
                print(f"[GamePlotter2D] Worker error: {e}")

    def _process_update(self, update_data):
        """Process a single update (called from worker thread)."""
        if not self._started or not self._visible:
            return
        
        game_state, overlays = update_data
        
        with self._worker_lock:
            # Balls
            bxs = [b.x for b in game_state.balls]
            bys = [b.y for b in game_state.balls]
            self._balls_item.setData(x=bxs, y=bys, _callSync='off')

            # Players
            pxs = [p.x for p in game_state.players]
            pys = [p.y for p in game_state.players]
            self._players_item.setData(x=pxs, y=pys, _callSync='off')

            # Direction arrows
            L = self.arrow_length
            axs, ays = [], []
            for p in game_state.players:
                axs += [p.x, p.x + L * math.cos(p.angle), float('nan')]
                ays += [p.y, p.y + L * math.sin(p.angle), float('nan')]
            self._arrows_item.setData(x=axs, y=ays, _callSync='off')
            
            # Player ID text labels
            rpg = self.rpg
            for i, p in enumerate(game_state.players):
                # Create text item if needed
                if i >= len(self._player_text_items):
                    text_item = rpg.TextItem(
                        anchor=(0.5, 0.5),
                        color='y'
                    )
                    self._win.addItem(text_item, _callSync='off')
                    self._player_text_items.append(text_item)
                
                # Update text and position with larger font (3x default ~12pt = 36pt)
                text_item = self._player_text_items[i]
                text_item.setHtml(
                    f'<span style="font-size: 12pt; color: white; font-weight: bold;">{p.id}</span>',
                    _callSync='off'
                )
                text_item.setPos(p.x, p.y, _callSync='off')
                text_item.setVisible(True, _callSync='off')
            
            # Hide unused text items
            for i in range(len(game_state.players), len(self._player_text_items)):
                self._player_text_items[i].setVisible(False, _callSync='off')
            
            # Overlays (external drawing updates)
            for overlay_item, data in overlays:
                overlay_item.setData(**data, _callSync='off')

    def update(self, game_state, overlays=None):
        """Push new game state to the remote plot. Never blocks the caller.

        If the worker thread is busy, this drops the old queued update and
        replaces it with this new one (always shows most recent state).

        Args:
            game_state: Object with .balls (list of objects with .x/.y) and
                        .players (list of objects with .x/.y/.angle).
            overlays: Optional list of (overlay_item, data_dict) tuples.
                     Each data_dict is passed to overlay_item.setData().
        """
        if not self._started or not self._visible:
            return
        
        if overlays is None:
            overlays = []
        
        # Try to put update in queue, drop old one if full (non-blocking)
        try:
            # Clear old update if present
            try:
                self._update_queue.get_nowait()
            except queue.Empty:
                pass
            # Put new update
            self._update_queue.put_nowait((game_state, overlays))
        except queue.Full:
            # Should never happen with maxsize=1 and get_nowait above
            pass

    def check_click(self):
        """Poll for click events. Returns (x, y) in metres or None.

        Call this once per loop iteration to detect user clicks on the plot.
        Only returns the most recent click since the last call.
        """
        if not self._started:
            return None

        try:
            mtime = os.path.getmtime(self._click_file)
            if mtime != self._last_click_mtime:
                self._last_click_mtime = mtime
                with open(self._click_file) as f:
                    line = f.read().strip()
                if line:
                    x, y = map(float, line.split(','))
                    return (x, y)
        except (FileNotFoundError, ValueError, OSError):
            pass
        return None

    def add_overlay(self, **kwargs):
        """Create a custom overlay plot item for external drawing.

        Returns an ObjectProxy to a PlotDataItem in the child process.
        Call .setData(x=..., y=..., _callSync='off') on it each frame to update.

        Example:
            # Create a dashed cyan circle overlay
            rpg = plotter.rpg
            overlay = plotter.add_overlay(
                pen=rpg.mkPen('c', width=2, style=rpg.QtCore.Qt.DashLine)
            )

            # Update it each frame
            overlay.setData(x=circle_xs, y=circle_ys, _callSync='off')

        Args:
            **kwargs: Arguments passed to win.plot() in the child process.
                     Common args: x, y, pen, symbol, symbolSize, symbolBrush, symbolPen

        Returns:
            ObjectProxy to the PlotDataItem, or None if not started.
        """
        if not self._started:
            return None
        return self._win.plot(**kwargs)


# ---------------------------------------------------------------------------
# Standalone test with real board_config (includes background image)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import time
    from dataclasses import dataclass
    from board_config import global_board_config

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

    plotter = GamePlotter2D(global_board_config)
    plotter.start()

    bw, bh = global_board_config.get_board_dimensions()
    pw, ph = global_board_config.get_print_dimensions()

    # --- External overlay demonstration ---
    # Consumer code creates custom overlays using add_overlay()
    rpg = plotter.rpg
    overlay_circle = plotter.add_overlay(
        pen=rpg.mkPen('c', width=2, style=rpg.QtCore.Qt.DashLine)
    )
    overlay_marker = plotter.add_overlay(
        x=[0], y=[0],
        pen=None,
        symbol='+',
        symbolSize=15,
        symbolPen=rpg.mkPen('m', width=2),
    )

    def make_circle(cx, cy, r, n=16):  # Reduced from 32 to 16 points
        angles = [2 * math.pi * i / n for i in range(n + 1)]
        return [cx + r * math.cos(a) for a in angles], [cy + r * math.sin(a) for a in angles]

    t = 0.0
    print("Remote plotter running with board_config. Ctrl+C to quit.")
    print(f"Board dimensions: {bw:.3f} x {bh:.3f} m")
    print(f"Print dimensions: {pw:.3f} x {ph:.3f} m")
    print(f"Image: {global_board_config.image_path}")
    print("Click anywhere on the plot to see coordinates.")
    print("Watch for: cyan dashed circle (pulsing) and magenta cross (drifting)")
    print()
    print("Timing breakdown:")
    print("  [prep]   = data preparation (game state + overlays)")
    print("  [update] = update() call (just queues data, never blocks)")
    print("  [total]  = entire main-process loop iteration")
    print("Status: Worker thread handles all IPC in background")
    print()
    try:
        while True:
            loop_start = time.monotonic()

            # Scale motion to board dimensions
            # One ball on a figure-8 (Lissajous)
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
                    angle=t + math.pi / 2,      # tangent direction
                ),
                DummyPlayer(
                    id=1,
                    x=r * math.cos(t + math.pi),
                    y=r * math.sin(t + math.pi),
                    angle=t + math.pi * 3 / 2,
                ),
            ]

            state = DummyGameState(balls, players)
            
            # Prepare overlay data
            circle_r = 0.1 + 0.05 * math.sin(t * 2)
            cxs, cys = make_circle(0, 0, circle_r)
            marker_x = 0.15 * math.sin(t * 1.5)
            marker_y = 0.10 * math.cos(t * 1.5)
            
            prep_time = (time.monotonic() - loop_start) * 1000
            update_start = time.monotonic()
            
            # Single non-blocking update with overlays
            plotter.update(state, overlays=[
                (overlay_circle, {'x': cxs, 'y': cys}),
                (overlay_marker, {'x': [marker_x], 'y': [marker_y]}),
            ])
            
            update_time = (time.monotonic() - update_start) * 1000
            
            # Check for clicks (file-based, no IPC needed)
            click = plotter.check_click()
            if click is not None:
                print(f"  [CLICK] ({click[0]:.3f}, {click[1]:.3f})")
            
            total_time = (time.monotonic() - loop_start) * 1000
            
            print(f"[prep] {prep_time:.2f} ms  [update] {update_time:.2f} ms  [total] {total_time:.2f} ms")

            t += 0.05
            time.sleep(0.02)

    except KeyboardInterrupt:
        pass
    finally:
        plotter.stop()
        print("Done.")
