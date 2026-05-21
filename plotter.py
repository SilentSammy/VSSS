"""
plotter.py — visualization loop.

ZMQ topology:
  CONNECT SUB tcp://localhost:5556  ← receives PlotGameState from main
  BIND PUB    tcp://*:5557          → publishes ClickEvents to main
"""

import time
import os
import numpy as np
import zmq
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import cv2

from zmq_comms import PlotGameState, ClickEvent, PlotUpdate


class PlotOverlay:
    """A group of animated artists managed together for blitting."""

    def __init__(self, plotter, artists):
        self.plotter = plotter
        self.artists = list(artists)

    def set_artists(self, new_artists):
        for a in self.artists:
            try:
                a.remove()
            except Exception:
                pass
        self.artists = list(new_artists)

    def clear(self):
        self.set_artists([])

    def remove(self):
        self.set_artists([])
        try:
            self.plotter._overlays.remove(self)
        except ValueError:
            pass


class GamePlotter2D:
    """Real-time 2D plotting of game state using matplotlib with blitting for performance."""

    def __init__(self, board_config=None, field_width=1.0, field_height=0.7, figsize=(8, 6),
                 ball_color='orange', ball_radius=0.01,
                 player_color='blue', player_alpha=0.7, player_length=0.02,
                 text_color='white', text_size=7, margin=0.2, on_click=None):
        self.board_config = board_config
        self.margin = margin
        self.figsize = figsize
        self.on_click = on_click
        self.ball_color = ball_color
        self.ball_radius = ball_radius
        self.player_color = player_color
        self.player_alpha = player_alpha
        self.player_length = player_length
        self.text_color = text_color
        self.text_size = text_size
        self.fig = None
        self.ax = None
        self.background = None
        self._overlays = []
        self._balls_overlay = None
        self._players_overlay = None
        self._svg_overlay = None
        self._waypoint_overlay = None
        self._sticky_svg_points = None
        self._sticky_waypoint = None

    def add_overlay(self, artists):
        overlay = PlotOverlay(self, artists)
        self._overlays.append(overlay)
        return overlay

    def start(self):
        if self.board_config is not None:
            self.field_width, self.field_height = self.board_config.get_board_dimensions()
            print_width, print_height = self.board_config.get_print_dimensions()
        else:
            self.field_width, self.field_height = 1.0, 0.7
            print_width, print_height = self.field_width, self.field_height

        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.fig.canvas.manager.set_window_title('VSS Game State')
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('VSS Game State')
        self.ax.grid(True, alpha=0.3)

        if self.board_config is not None and os.path.exists(self.board_config.image_path):
            img = cv2.imread(self.board_config.image_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.ax.imshow(img_rgb,
                               extent=(-print_width / 2, print_width / 2,
                                       -print_height / 2, print_height / 2),
                               aspect='auto', zorder=0)

        paper_rect = Rectangle(
            (-print_width / 2, -print_height / 2),
            print_width, print_height,
            fill=False, edgecolor='red', linewidth=2, zorder=1
        )
        self.ax.add_patch(paper_rect)

        self.ax.set_xlim(-print_width / 2 - self.margin, print_width / 2 + self.margin)
        self.ax.set_ylim(-print_height / 2 - self.margin, print_height / 2 + self.margin)

        self._svg_overlay = self.add_overlay([])
        self._waypoint_overlay = self.add_overlay([])
        self._balls_overlay = self.add_overlay([])
        self._players_overlay = self.add_overlay([])

        self.fig.show()
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

        self.fig.canvas.mpl_connect('resize_event', self._on_resize)
        self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_click)

    def _on_resize(self, event):
        self.ax.set_aspect('equal', adjustable='box')
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def _on_mouse_click(self, event):
        if event.button == 1 and event.inaxes == self.ax and self.on_click is not None:
            self.on_click(event.xdata, event.ydata)

    def _rebuild_game_objects(self, game_state):
        ball_artists = []
        for ball in game_state.balls:
            c = Circle((ball.x, ball.y), radius=self.ball_radius,
                       color=self.ball_color, zorder=10)
            self.ax.add_patch(c)
            ball_artists.append(c)
        self._balls_overlay.set_artists(ball_artists)

        player_artists = []
        for player in game_state.players:
            r = self.player_length
            c = Circle((player.x, player.y), radius=r,
                       color=self.player_color, alpha=self.player_alpha, zorder=10)
            self.ax.add_patch(c)
            player_artists.append(c)
            tip_x = player.x + r * 1.5 * np.cos(player.angle)
            tip_y = player.y + r * 1.5 * np.sin(player.angle)
            line, = self.ax.plot([player.x, tip_x], [player.y, tip_y],
                                 'k-', linewidth=2, zorder=11)
            player_artists.append(line)
            text = self.ax.text(
                player.x, player.y, str(player.id),
                ha='center', va='center',
                fontsize=self.text_size, color=self.text_color,
                weight='bold', zorder=12
            )
            player_artists.append(text)
        self._players_overlay.set_artists(player_artists)

    def _rebuild_overlays(self, game_state):
        # SVG path
        svg_artists = []
        if self._sticky_svg_points and len(self._sticky_svg_points) > 1:
            xs = [p[0] for p in self._sticky_svg_points]
            ys = [p[1] for p in self._sticky_svg_points]
            line, = self.ax.plot(xs, ys, color='red', linewidth=1.5, zorder=5, alpha=0.8)
            svg_artists.append(line)
        self._svg_overlay.set_artists(svg_artists)

        # Waypoint
        wp_artists = []
        if self._sticky_waypoint:
            wx, wy = self._sticky_waypoint
            dot, = self.ax.plot(wx, wy, 'x', color='magenta', markersize=10,
                                markeredgewidth=2, zorder=13)
            wp_artists.append(dot)
            if game_state.players:
                p0 = game_state.players[0]
                link, = self.ax.plot([p0.x, wx], [p0.y, wy],
                                     '--', color='magenta', linewidth=1.5, zorder=12, alpha=0.7)
                wp_artists.append(link)
        self._waypoint_overlay.set_artists(wp_artists)

    def update(self, plot_update):
        game_state = plot_update.game_state

        if plot_update.svg_points is not None:
            self._sticky_svg_points = plot_update.svg_points or None
        if plot_update.waypoint is not None:
            self._sticky_waypoint = plot_update.waypoint or None

        self._rebuild_overlays(game_state)
        self._rebuild_game_objects(game_state)

        self.fig.canvas.restore_region(self.background)
        for overlay in self._overlays:
            for artist in overlay.artists:
                self.ax.draw_artist(artist)
        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()

    def close(self):
        plt.close(self.fig)


GAME_STATE_PORT = 5556
CLICK_PORT      = 5557


if __name__ == '__main__':
    from board_config import global_board_config

    context = zmq.Context()

    # Inbound: game state from main
    sub = context.socket(zmq.SUB)
    sub.setsockopt(zmq.CONFLATE, 1)   # keep only the latest message (no backlog)
    sub.connect(f"tcp://localhost:{GAME_STATE_PORT}")
    sub.setsockopt_string(zmq.SUBSCRIBE, '')

    # Outbound: click events to main
    pub = context.socket(zmq.PUB)
    pub.bind(f"tcp://*:{CLICK_PORT}")

    print(f"[plotter] Subscribing to game state on port {GAME_STATE_PORT}")
    print(f"[plotter] Publishing clicks on port {CLICK_PORT}")

    def on_click(x, y):
        click = ClickEvent(x=x, y=y, timestamp=time.time())
        pub.send_string(click.to_json())
        print(f"\n[plotter] Click sent: ({x:.3f}, {y:.3f})")

    plotter = GamePlotter2D(
        board_config=global_board_config,
        player_length=0.075,
        on_click=on_click,
    )
    plotter.start()
    plt.pause(0.5)

    running = True
    plotter.fig.canvas.mpl_connect('close_event', lambda e: globals().update(running=False))

    last_state = PlotUpdate(game_state=PlotGameState())
    last_time  = time.perf_counter()

    try:
        while running:
            try:
                try:
                    msg = sub.recv_string(zmq.NOBLOCK)
                    last_state = PlotUpdate.from_json(msg)
                except zmq.Again:
                    pass

                plotter.update(last_state)

                now = time.perf_counter()
                dt  = now - last_time
                last_time = now
                print(f"\r[plotter] Plot loop: {1/dt:.1f} Hz   ", end='', flush=True)

                time.sleep(0.01)
            except KeyboardInterrupt:
                break
            except Exception as e:
                import traceback
                print(f"\n[plotter] ERROR: {e}")
                traceback.print_exc()
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("\n[plotter] Stopped")
        plotter.close()
        sub.close()
        pub.close()
        context.term()
