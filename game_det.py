from dataclasses import dataclass
from typing import List
import numpy as np
from obj_det import ArucoDetector, BallDetector
from board_est import BoardEstimator
import cv2
import board_config
from board_config import global_board_config
from cam_config import global_cam

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

class GamePlotter2D:
    """Real-time 2D plotting of game state using matplotlib with blitting for performance."""
    
    def __init__(self, board_config, figsize=(8, 6),
                 ball_color='orange', ball_radius=0.01,
                 player_color='blue', player_alpha=0.7, player_length=0.02, player_width=0.015,
                 text_color='white', text_size=7):
        """Initialize 2D plotter.
        
        Args:
            board_config: BoardConfig instance for field dimensions
            figsize: Figure size in inches (width, height)
            ball_color: Ball circle fill color
            ball_radius: Ball circle radius in meters
            player_color: Player triangle fill color
            player_alpha: Player triangle transparency (0-1)
            player_length: Player triangle length in meters
            player_width: Player triangle base width in meters
            text_color: Player ID text color
            text_size: Player ID text font size
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        import cv2
        import os
        
        self.board_config = board_config
        self.field_width, self.field_height = board_config.get_board_dimensions()
        print_width, print_height = board_config.get_print_dimensions()
        
        # Appearance settings
        self.ball_color = ball_color
        self.ball_radius = ball_radius
        self.player_color = player_color
        self.player_alpha = player_alpha
        self.player_length = player_length
        self.player_width = player_width
        self.text_color = text_color
        self.text_size = text_size
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.fig.canvas.manager.set_window_title('VSS Game State')
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('VSS Game State')
        self.ax.grid(True, alpha=0.3)
        
        # Load and display board image as background
        if os.path.exists(board_config.image_path):
            img = cv2.imread(board_config.image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Display image centered, scaled to print dimensions
            self.ax.imshow(img_rgb, extent=(-print_width/2, print_width/2, -print_height/2, print_height/2),
                          aspect='auto', zorder=0)
        
        # Draw paper border rectangle
        paper_rect = Rectangle(
            (-print_width/2, -print_height/2), 
            print_width, print_height,
            fill=False, edgecolor='red', linewidth=2, zorder=1
        )
        self.ax.add_patch(paper_rect)
        
        # Set limits with margin
        margin = 0.05
        self.ax.set_xlim(-print_width/2 - margin, print_width/2 + margin)
        self.ax.set_ylim(-print_height/2 - margin, print_height/2 + margin)
        
        # Initialize plot artists (animated=True for blitting)
        self.ball_artists = []
        self.player_artists = []
        self.player_text_artists = []
        
        # Show and draw once to initialize
        self.fig.show()
        self.fig.canvas.draw()
        
        # Capture background for blitting
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        
        # Connect resize event to recapture background
        self.fig.canvas.mpl_connect('resize_event', self._on_resize)
        
        # Trigger resize to fix initial aspect ratio
        self._on_resize(None)
    
    def _on_resize(self, event):
        """Handle window resize by recapturing background."""
        self.ax.set_aspect('equal', adjustable='box')
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
    
    def update(self, game_state):
        """Update plot with new game state using blitting.
        
        Args:
            game_state: GameState object with balls and players
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Polygon
        
        # Restore background (clears old artists)
        self.fig.canvas.restore_region(self.background)
        
        # Remove old artists
        for artist in self.ball_artists + self.player_artists + self.player_text_artists:
            artist.remove()
        self.ball_artists.clear()
        self.player_artists.clear()
        self.player_text_artists.clear()
        
        # Draw balls
        if game_state.balls:
            for ball in game_state.balls:
                circle = Circle(
                    (ball.x, ball.y), 
                    radius=self.ball_radius,
                    color=self.ball_color,
                    animated=True,
                    zorder=10
                )
                self.ax.add_patch(circle)
                self.ball_artists.append(circle)
                self.ax.draw_artist(circle)
        
        # Draw players
        if game_state.players:
            for player in game_state.players:
                # Player triangle pointing in direction of angle
                # Triangle centroid centered at (player.x, player.y)
                length = self.player_length
                width = self.player_width
                
                # Direction vector
                cos_a = np.cos(player.angle)
                sin_a = np.sin(player.angle)
                
                # Triangle centroid at player position
                # Base center is 1/3 of length behind centroid
                # Tip is 2/3 of length ahead of centroid
                base_cx = player.x - (length / 3) * cos_a
                base_cy = player.y - (length / 3) * sin_a
                tip_x = player.x + (2 * length / 3) * cos_a
                tip_y = player.y + (2 * length / 3) * sin_a
                
                # Base corners perpendicular to direction
                base_angle = player.angle + np.pi / 2
                base1_x = base_cx + (width / 2) * np.cos(base_angle)
                base1_y = base_cy + (width / 2) * np.sin(base_angle)
                base2_x = base_cx - (width / 2) * np.cos(base_angle)
                base2_y = base_cy - (width / 2) * np.sin(base_angle)
                
                triangle = Polygon(
                    [[tip_x, tip_y], [base1_x, base1_y], [base2_x, base2_y]],
                    color=self.player_color,
                    alpha=self.player_alpha,
                    animated=True,
                    zorder=10
                )
                self.ax.add_patch(triangle)
                self.player_artists.append(triangle)
                self.ax.draw_artist(triangle)
                
                # Player ID text
                text = self.ax.text(
                    player.x, player.y,
                    f'{player.id}',
                    ha='center', va='center',
                    fontsize=self.text_size,
                    color=self.text_color,
                    animated=True,
                    zorder=11
                )
                self.player_text_artists.append(text)
                self.ax.draw_artist(text)
        
        # Blit the changes
        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()
    
    def close(self):
        """Close the plot window."""
        import matplotlib.pyplot as plt
        plt.close(self.fig)

# Setup-specific settings
is_small_setup = global_board_config == board_config.board_config_letter

# Setup GameDetector
game_detector = GameDetector(
    board_estimator=BoardEstimator(global_board_config, K=global_cam.K, D=global_cam.D, rotate_180=True),
    ball_detector=BallDetector(),
    ball_height=0.02,
    aruco_detector=ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)),
    player_height=0.04 if is_small_setup else 0.08,
)
if __name__ == "__main__":
    
    # Setup 2D plotter
    plotter = GamePlotter2D(
        global_board_config,
        player_width=0.015 if is_small_setup else 0.05,
        player_length=0.02 if is_small_setup else 0.075
    )
    
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
        plotter.close()
        cv2.destroyAllWindows()

