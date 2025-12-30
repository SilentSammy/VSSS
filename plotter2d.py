import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cv2
from board_config import board_config_letter
from board_est import BoardEstimator
from obj_det import BallDetector, ArucoDetector
from obj_loc import ObjectLocalizer
from cam_config import global_cam

matplotlib.use('TkAgg')


class BoardPlotter2D:
    """2D top-down view of board with ball position."""
    
    def __init__(self, board_config, update_interval=10,
                 ball_color='cyan', ball_size=400, ball_edge_color='black', ball_edge_width=2,
                 player_color='red', player_edge_color='black', player_edge_width=2,
                 player_size=0.02, player_width=0.02):
        """Initialize plotter.
        
        Args:
            board_config: Board configuration
            update_interval: Update plot every N frames
            ball_color: Ball marker fill color
            ball_size: Ball marker size in points
            ball_edge_color: Ball marker edge color
            ball_edge_width: Ball marker edge width
            player_color: Player triangle fill color
            player_edge_color: Player triangle edge color
            player_edge_width: Player triangle edge width
            player_size: Player triangle length (m)
            player_width: Player triangle base width (m)
        """
        self.config = board_config
        self.update_interval = update_interval
        self.frame_count = 0
        
        # Ball appearance
        self.ball_color = ball_color
        self.ball_size = ball_size
        self.ball_edge_color = ball_edge_color
        self.ball_edge_width = ball_edge_width
        
        # Player appearance
        self.player_color = player_color
        self.player_edge_color = player_edge_color
        self.player_edge_width = player_edge_width
        self.player_size = player_size
        self.player_width = player_width
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.fig.canvas.manager.set_window_title('Board 2D View')
        
        # Initialize board outline (will be drawn once)
        self._draw_board_outline()
        
        # Ball scatter plot (will be updated)
        self.ball_scatter = self.ax.scatter([], [], c=self.ball_color, s=self.ball_size, marker='o', 
                                           edgecolors=self.ball_edge_color, linewidths=self.ball_edge_width, zorder=10)
        
        # Player triangle (will be updated)
        self.player_triangle = None
        
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('Ball Position on Board')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        
        plt.ion()
        plt.show()
    
    def _draw_board_outline(self):
        """Draw board boundary with background image."""
        w, h = self.config.get_board_dimensions()
        print_w, print_h = self.config.get_print_dimensions()
        
        # Load and display board image as background
        import cv2
        import os
        if os.path.exists(self.config.image_path):
            img = cv2.imread(self.config.image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Display image centered, scaled to print dimensions
            self.ax.imshow(img_rgb, extent=(-print_w/2, print_w/2, -print_h/2, print_h/2),
                          aspect='auto', zorder=0)
        
        # Draw paper edge rectangle
        paper_rect = plt.Rectangle((-print_w/2, -print_h/2), print_w, print_h, 
                                   fill=False, edgecolor='red', linewidth=2, zorder=1)
        self.ax.add_patch(paper_rect)
        
        # Set limits with margin
        margin = 0.05
        self.ax.set_xlim(-print_w/2 - margin, print_w/2 + margin)
        self.ax.set_ylim(-print_h/2 - margin, print_h/2 + margin)
    
    def update(self, ball_pos, player_pos=None):
        """Update ball and player positions.
        
        Args:
            ball_pos: (x, y) tuple in board coordinates, or None
            player_pos: (x, y, angle_rad) tuple in board coordinates, or None
        """
        self.frame_count += 1
        
        if self.frame_count % self.update_interval != 0:
            return
        
        if ball_pos is not None:
            x, y = ball_pos[:2]
            self.ball_scatter.set_offsets([[x, y]])
        else:
            # Hide ball by setting empty array with proper shape
            self.ball_scatter.set_offsets(np.empty((0, 2)))
        
        if player_pos is not None:
            x, y, angle_rad = player_pos
            # Create triangle pointing in direction of player orientation
            triangle_length = self.player_size
            triangle_width = self.player_width
            
            # Define triangle vertices in local coordinates (pointing right)
            local_vertices = np.array([
                [triangle_length, 0],           # Tip
                [-triangle_length/3, triangle_width/2],   # Base top
                [-triangle_length/3, -triangle_width/2]   # Base bottom
            ])
            
            # Rotation matrix
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            # Rotate and translate vertices
            rotated_vertices = local_vertices @ rotation.T
            world_vertices = rotated_vertices + np.array([x, y])
            
            # Remove old triangle and create new one
            if self.player_triangle is not None:
                self.player_triangle.remove()
            
            self.player_triangle = plt.Polygon(world_vertices, color=self.player_color, 
                                              edgecolor=self.player_edge_color, linewidth=self.player_edge_width, zorder=11)
            self.ax.add_patch(self.player_triangle)
        else:
            # Hide triangle by removing it
            if self.player_triangle is not None:
                self.player_triangle.remove()
                self.player_triangle = None
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


if __name__ == "__main__":
    import cv2
    from cam_config import global_cam
    
    # Setup
    be = BoardEstimator(board_config_letter, K=global_cam.K, D=global_cam.D, rotate_180=True)
    ball_localizer = ObjectLocalizer(BallDetector(), be, height=0.02)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    player_localizer = ObjectLocalizer(ArucoDetector(aruco_dict, marker_id=12), be, height=0.04)
    plotter = BoardPlotter2D(board_config_letter, update_interval=5)
    
    while True:
        if cv2.waitKey(1) & 0xFF == 27:
            break
        
        frame = global_cam.get_frame()
        if frame is None:
            continue
        
        drawing_frame = frame.copy()
        
        # Detect board
        result = be.get_board_transform(frame, drawing_frame=drawing_frame)
        
        ball_pos = None
        player_pos = None
        if result is not None:
            board_T, pnp_result = result
            ball_pos = ball_localizer.localize(frame, pnp_result, drawing_frame)
            
            # Get player position and angle separately
            player_xyz = player_localizer.localize(frame, pnp_result, drawing_frame)
            if player_xyz is not None:
                # Get angle from ArUco detector
                _, contour = player_localizer.detector.detect(frame)
                if contour is not None:
                    angle_image = player_localizer.detector.get_angle(contour)
                    if angle_image is not None:
                        # Get camera rotation in board frame to offset ArUco angle
                        board_T = pnp_result.get_board_T()
                        cam_T_in_board = np.linalg.inv(board_T)
                        cam_R = cam_T_in_board[:3, :3]
                        
                        # Extract Z rotation (gamma) from camera orientation
                        from matrix_help import extract_euler_zyx
                        alpha, beta, gamma = extract_euler_zyx(cam_R)
                        
                        # Offset angle by camera's Z rotation
                        angle_board = angle_image - gamma + np.pi
                        player_pos = (player_xyz[0], player_xyz[1], angle_board)
            
            if ball_pos is not None:
                x, y, z = ball_pos
                cv2.putText(drawing_frame, f"Ball: ({x:.3f}, {y:.3f}, {z:.3f})m", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            if player_pos is not None:
                x, y, angle_rad = player_pos
                angle_deg = np.degrees(angle_rad)
                cv2.putText(drawing_frame, f"Player #12: ({x:.3f}, {y:.3f})m, {angle_deg:.1f}deg", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Update plotter
        plotter.update(ball_pos, player_pos)
        
        # Display frame
        cv2.imshow("Camera View", drawing_frame)
        cv2.setWindowProperty("Camera View", cv2.WND_PROP_TOPMOST, 1)
