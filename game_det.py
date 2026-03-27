from dataclasses import dataclass
from typing import List
import numpy as np
from obj_det import ArucoDetector, BallDetector
from board_est import BoardEstimator
import cv2
import board_config
from board_config import global_board_config
from cam_config import global_cam
from plotter2d_remote import GamePlotter2D

@dataclass
class GameState:
    """Represents the current state of all game objects."""
    balls: List = None      # List of BallState objects
    players: List = None    # List of PlayerState objects
    board_transform: np.ndarray = None  # 4x4 transformation matrix from board to camera
    pnp_result: tuple = None  # PnP solution (rvec, tvec, etc.)
    detector: 'GameDetector' = None  # Reference to parent GameDetector
    timestamp: float = None  # Detection timestamp
    
    def __post_init__(self):
        if self.balls is None:
            self.balls = []
        if self.players is None:
            self.players = []

    def get_player(self, aruco_id):
        return next((p for p in self.players if p.id == aruco_id), None)

@dataclass
class BallState:
    """Represents a single ball's state."""
    x: float
    y: float
    detection: 'DetectedObject' = None  # Original detection object

@dataclass
class PlayerState:
    """Represents a single player's state."""
    id: int
    x: float
    y: float
    angle: float
    detection: 'DetectedAruco' = None  # Original detection object

class GameDetector:
    """Detects game objects and returns GameState."""
    
    def __init__(self, board_estimator, ball_detector=None, ball_height=0.0, 
                 aruco_detector=None, player_height=0.0):
        """Initialize game detector.
        
        Args:
            board_estimator: BoardEstimator instance for detecting board pose
            ball_detector: Optional ObjectDetector for ball detection
            ball_height: Ball height above board (m) for parallax correction
            aruco_detector: Optional ObjectDetector for player ArUco marker detection
            player_height: Player height above board (m) for parallax correction
        """
        self.board_estimator = board_estimator

        self.ball_detector = ball_detector or BallDetector()
        self.ball_height = ball_height

        self.player_detector = aruco_detector
        self.player_height = player_height
    
    @staticmethod
    def _same_marker_size(dict1, dict2):
        """Check if two ArUco dictionaries have the same marker dimensions.
        
        Args:
            dict1: First cv2.aruco.Dictionary
            dict2: Second cv2.aruco.Dictionary
            
        Returns:
            bool: True if both have same marker size (e.g., both 4x4, both 5x5)
        """
        if dict1 is None or dict2 is None:
            return False
        return dict1.markerSize == dict2.markerSize

    def _localize(self, frame, centroid, pnp_result, height):
        """Detect object and return 3D board coordinates with parallax correction."""
        
        if centroid is None:
            return None
        
        x, y = self.board_estimator.project_point_to_board(
            pnp_result, centroid, frame.shape, z=height
        )
        
        return (x, y, height)
    
    def detect(self, frame, drawing_frame=None):
        """Detect all game objects and return GameState.
        
        Args:
            frame: Input image frame
            drawing_frame: Optional frame to draw detections on
            
        Returns:
            GameState with detected balls and players
        """
        import time
        
        # Detect board first
        result = self.board_estimator.get_board_transform(frame)
        
        if result is None:
            return GameState()  # Return empty state if board not detected
        
        board_T, pnp_result = result
        timestamp = time.time()
        
        balls = []
        players = []
        
        # Detect balls
        if self.ball_detector is not None:
            ball_detections = self.ball_detector.detect(frame)
            for ball in ball_detections:
                xyz = self._localize(frame, ball.centroid, pnp_result, self.ball_height)
                if xyz is not None:
                    balls.append(BallState(
                        x=xyz[0],
                        y=xyz[1],
                        detection=ball
                    ))
                    
                    # Draw ball contour
                    if drawing_frame is not None and ball.contour is not None:
                        cv2.drawContours(drawing_frame, [ball.contour], -1, (0, 255, 255), 2)
        
        # Detect players
        if self.player_detector is not None:
            player_detections = self.player_detector.detect(frame)
            for player in player_detections:
                # Skip board markers (only if same marker dimensions)
                if (self.board_estimator.config.board_marker_ids is not None and 
                    player.id in self.board_estimator.config.board_marker_ids and
                    self._same_marker_size(player.dict, self.board_estimator.config.dictionary)):
                    continue
                
                xyz = self._localize(frame, player.centroid, pnp_result, self.player_height)
                if xyz is not None and player.angle is not None:
                    # Transform angle from image space to board space
                    # Get camera rotation in board frame
                    cam_T_in_board = np.linalg.inv(board_T)
                    cam_R = cam_T_in_board[:3, :3]
                    
                    # Extract Z rotation (gamma) from camera orientation
                    from matrix_help import extract_euler_zyx
                    alpha, beta, gamma = extract_euler_zyx(cam_R)
                    
                    # Transform angle
                    angle_board = (player.angle - gamma + np.pi) % (2 * np.pi)
                              
                    players.append(PlayerState(
                        id=player.id,
                        x=xyz[0],
                        y=xyz[1],
                        angle=angle_board,
                        detection=player
                    ))
                    
                    # Draw player triangle using image-space angle
                    if drawing_frame is not None and player.centroid is not None:
                        cx, cy = int(player.centroid[0]), int(player.centroid[1])
                        
                        # Triangle size
                        length = 20
                        width = 12
                        
                        # Direction vector (flipped angle for image space)
                        cos_a = np.cos(-player.angle)
                        sin_a = np.sin(-player.angle)
                        
                        # Triangle centroid at marker position
                        # Base center is 1/3 of length behind centroid
                        # Tip is 2/3 of length ahead of centroid
                        base_cx = cx - (length / 3) * cos_a
                        base_cy = cy - (length / 3) * sin_a
                        tip_x = cx + (2 * length / 3) * cos_a
                        tip_y = cy + (2 * length / 3) * sin_a
                        
                        # Base corners perpendicular to direction
                        base_angle = -player.angle + np.pi / 2
                        base1_x = base_cx + (width / 2) * np.cos(base_angle)
                        base1_y = base_cy + (width / 2) * np.sin(base_angle)
                        base2_x = base_cx - (width / 2) * np.cos(base_angle)
                        base2_y = base_cy - (width / 2) * np.sin(base_angle)
                        
                        # Draw filled triangle
                        pts = np.array([[tip_x, tip_y], [base1_x, base1_y], [base2_x, base2_y]], np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.fillPoly(drawing_frame, [pts], (255, 100, 255))
        
        return GameState(
            balls=balls,
            players=players,
            board_transform=board_T,
            pnp_result=pnp_result,
            detector=self,
            timestamp=timestamp
        )

class PlotOverlay:
    """Handle to a group of animated artists managed by GamePlotter2D.

    Obtain via plotter.add_overlay(artists).
    Artists must already be added to plotter.ax before calling add_overlay.
    Call remove() to deregister and clean up all artists.

    Example:
        line, = plotter.ax.plot([], [], 'r--', zorder=5)
        overlay = plotter.add_overlay([line])
        # each frame before plotter.update():
        line.set_data(xs, ys)
        # when no longer needed:
        overlay.remove()
    """

    def __init__(self, plotter, artists):
        self._plotter = plotter
        self._artists = []
        self._apply(artists)

    def _apply(self, artists):
        for a in self._artists:
            try:
                a.remove()
            except Exception:
                pass
        self._artists = list(artists)
        for a in self._artists:
            a.set_animated(True)

    @property
    def artists(self):
        return self._artists

    def set_artists(self, new_artists):
        """Replace artists. Old ones are removed from axes; new ones must already be added."""
        self._apply(new_artists)

    def remove(self):
        """Remove all artists from axes and deregister from plotter."""
        self._apply([])
        try:
            self._plotter._overlays.remove(self)
        except ValueError:
            pass

class PathOverlay:
    """Animated overlay that draws a sequence of points with a pursuit indicator.

    Shows:
    - Highlighted dot + pursuit line from car to the first (active) point
    - Dashed line + dots through all remaining points

    Usage:
        overlay = PathOverlay(plotter)
        # each frame:
        overlay.update(points, player_pos=(x, y))
    """

    def __init__(self, plotter,
                 path_color='c', path_lw=1.5, path_ms=6, path_style='--',
                 target_color='y', target_ms=8, pursuit_lw=1.5,
                 start_color='lime', start_ms=10):
        ax = plotter.ax
        self._pursuit_line, = ax.plot([], [], '-',        color=target_color, lw=pursuit_lw, zorder=5)
        self._target_dot,   = ax.plot([], [], 'o',        color=target_color, ms=target_ms,  zorder=6)
        self._path_line,    = ax.plot([], [], path_style, color=path_color,   lw=path_lw,    zorder=5)
        self._path_dots,    = ax.plot([], [], 'o',        color=path_color,   ms=path_ms,    zorder=5)
        self._start_dot,    = ax.plot([], [], 'o',        color=start_color,  ms=start_ms,   zorder=7,
                                      markerfacecolor='none', markeredgewidth=2)
        plotter.add_overlay([self._path_line, self._path_dots,
                             self._pursuit_line, self._target_dot, self._start_dot])

    def update(self, points, player_pos=None, start_point=None):
        """Redraw the overlay.

        Args:
            points: list of (x, y); first point is the active target.
            player_pos: (x, y) of the car, or None to hide the pursuit line.
            start_point: (x, y) to highlight as the path start, or None to hide.
        """
        if not points:
            for artist in (self._pursuit_line, self._target_dot,
                           self._path_line, self._path_dots, self._start_dot):
                artist.set_data([], [])
            return

        if start_point is not None:
            self._start_dot.set_data([start_point[0]], [start_point[1]])
        else:
            self._start_dot.set_data([], [])

        tx, ty = points[0]
        self._target_dot.set_data([tx], [ty])

        if player_pos is not None:
            self._pursuit_line.set_data([player_pos[0], tx], [player_pos[1], ty])
        else:
            self._pursuit_line.set_data([], [])

        rest = points[1:]
        if rest:
            xs, ys = zip(*points)        # line through active + queued
            self._path_line.set_data(xs, ys)
            rxs, rys = zip(*rest)        # dots only on queued
            self._path_dots.set_data(rxs, rys)
        else:
            self._path_line.set_data([], [])
            self._path_dots.set_data([], [])

# Setup-specific settings
is_small_setup = global_board_config == board_config.board_config_letter

# Setup GameDetector
game_detector = GameDetector(
    board_estimator=BoardEstimator(global_board_config, K=global_cam.K, D=global_cam.D, rotate_180=True),
    ball_detector=BallDetector(),
    ball_height=0.02,
    aruco_detector=ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)),
    player_height=0.04 if is_small_setup else 0.1,
)

# Setup GamePlotter2D
global_plotter = GamePlotter2D(global_board_config)

if __name__ == "__main__":
    plotter = global_plotter
    # Setup 2D plotter
    plotter.start()  # Create plot window
    
    try:
        while True:
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break
            
            frame = global_cam.get_frame()
            
            if frame is None:
                continue
            
            drawing_frame = frame.copy()
            
            # Detect game state
            game_state = game_detector.detect(frame, drawing_frame)
            game_state.balls = []
            
            # Update 2D plot
            plotter.update(game_state)
            
            # Annotate camera view with ball positions
            for i, ball in enumerate(game_state.balls):
                cv2.putText(drawing_frame, f"Ball: ({ball.x:.3f}, {ball.y:.3f})m",
                        (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Annotate camera view with player positions
            for i, player in enumerate(game_state.players):
                angle_deg = np.degrees(player.angle)
                cv2.putText(drawing_frame, f"Player {player.id}: ({player.x:.3f}, {player.y:.3f})m, {angle_deg:.1f}deg",
                           (10, 60 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 255), 2)
            
            # Display camera view
            cv2.imshow("Game Detection", drawing_frame)
            cv2.setWindowProperty("Game Detection", cv2.WND_PROP_TOPMOST, 1)
    finally:
        plotter.stop()
        cv2.destroyAllWindows()

