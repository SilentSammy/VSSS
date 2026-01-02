import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cv2
from board_config import board_config_letter
from board_est import BoardEstimator
from obj_det import BallDetector, ArucoDetector
from obj_loc import ObjectLocalizer
from cam_config import global_cam
import threading
import queue

matplotlib.use('Qt5Agg')


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
        
        # Early return without drawing if not updating this frame
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
            
            # Update existing triangle or create new one
            if self.player_triangle is None:
                self.player_triangle = plt.Polygon(world_vertices, facecolor=self.player_color, 
                                                  edgecolor=self.player_edge_color, linewidth=self.player_edge_width, zorder=11)
                self.ax.add_patch(self.player_triangle)
            else:
                # Just update vertices instead of removing/recreating
                self.player_triangle.set_xy(world_vertices)
        else:
            # Hide triangle by setting it off-screen
            if self.player_triangle is not None:
                self.player_triangle.set_xy(np.array([[1e6, 1e6], [1e6, 1e6], [1e6, 1e6]]))
        
        # Let matplotlib render asynchronously without blocking
        self.fig.canvas.draw_idle()


class ThreadedPlotter2D:
    """Non-blocking wrapper that runs detection loop in background, plotter in main thread."""
    
    def __init__(self, board_config, update_interval=5):
        """Initialize threaded plotter.
        
        Args:
            board_config: Board configuration
            update_interval: Update plot every N frames
        """
        self.position_queue = queue.Queue(maxsize=1)  # Only keep latest position
        self.board_config = board_config
        self.update_interval = update_interval
        self.plotter = BoardPlotter2D(board_config, update_interval=update_interval)
    
    def update(self, ball_pos, player_pos):
        """Update plotter with new positions.
        
        Args:
            ball_pos: (x, y) tuple in board coordinates, or None
            player_pos: (x, y, angle_rad) tuple in board coordinates, or None
        """
        self.plotter.update(ball_pos, player_pos)
    
    def poll_queue(self):
        """Check for position updates from detection thread (call from main thread).
        
        Returns:
            (ball_pos, player_pos) tuple if available, else (None, None)
        """
        try:
            return self.position_queue.get_nowait()
        except queue.Empty:
            return None, None
    
    def send_positions(self, ball_pos, player_pos):
        """Send positions from detection thread to main thread (non-blocking).
        
        Args:
            ball_pos: (x, y) tuple in board coordinates, or None
            player_pos: (x, y, angle_rad) tuple in board coordinates, or None
        """
        # Drop old positions, only keep latest
        if self.position_queue.full():
            try:
                self.position_queue.get_nowait()
            except queue.Empty:
                pass
        
        try:
            self.position_queue.put_nowait((ball_pos, player_pos))
        except queue.Full:
            pass  # Skip update if queue somehow full



if __name__ == "__main__":
    import cv2
    from cam_config import global_cam
    import time
    
    # Setup detection objects
    be = BoardEstimator(board_config_letter, K=global_cam.K, D=global_cam.D, rotate_180=True)
    ball_localizer = ObjectLocalizer(BallDetector(), be, height=0.02)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    player_localizer = ObjectLocalizer(ArucoDetector(aruco_dict, marker_id=12), be, height=0.04)
    
    # Create plotter in main thread (matplotlib requirement)
    plotter = ThreadedPlotter2D(board_config_letter, update_interval=5)
    
    # Shared state for detection thread
    stop_event = threading.Event()
    
    def detection_loop():
        """Run camera/detection in background thread."""
        while not stop_event.is_set():
            t0 = time.perf_counter()
            
            frame = global_cam.get_frame()
            t1 = time.perf_counter()
            print(f"[1] Get frame: {(t1-t0)*1000:.1f}ms")
            
            if frame is None:
                continue
            
            drawing_frame = frame.copy()
            
            # Detect board
            result = be.get_board_transform(frame, drawing_frame=drawing_frame)
            t2 = time.perf_counter()
            print(f"[2] Board transform: {(t2-t1)*1000:.1f}ms")
            
            ball_pos = None
            player_pos = None
            if result is not None:
                board_T, pnp_result = result
                ball_pos = ball_localizer.localize(frame, pnp_result, drawing_frame)
                t3 = time.perf_counter()
                print(f"[3] Ball localize: {(t3-t2)*1000:.1f}ms")
                
                # Get player position and angle separately
                player_xyz = player_localizer.localize(frame, pnp_result, drawing_frame)
                t4 = time.perf_counter()
                print(f"[4] Player localize: {(t4-t3)*1000:.1f}ms")
                
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
                    t5 = time.perf_counter()
                    print(f"[5] Player angle: {(t5-t4)*1000:.1f}ms")
                
                if ball_pos is not None:
                    x, y, z = ball_pos
                    cv2.putText(drawing_frame, f"Ball: ({x:.3f}, {y:.3f}, {z:.3f})m", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                if player_pos is not None:
                    x, y, angle_rad = player_pos
                    angle_deg = np.degrees(angle_rad)
                    cv2.putText(drawing_frame, f"Player #12: ({x:.3f}, {y:.3f})m, {angle_deg:.1f}deg", 
                               (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Send positions to main thread (non-blocking)
            plotter.send_positions(ball_pos, player_pos)
            t6 = time.perf_counter()
            print(f"[6] Send to plotter: {(t6-t2)*1000:.1f}ms")
            
            # Display frame
            cv2.imshow("Camera View", drawing_frame)
            cv2.setWindowProperty("Camera View", cv2.WND_PROP_TOPMOST, 1)
            
            # Check for ESC key to exit
            if cv2.waitKey(1) & 0xFF == 27:
                stop_event.set()
                break
            
            t7 = time.perf_counter()
            print(f"[7] CV display: {(t7-t6)*1000:.1f}ms")
            print(f"[TOTAL] Loop time: {(t7-t0)*1000:.1f}ms\n")
    
    # Start detection in background thread
    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()
    
    # Main thread: run matplotlib event loop
    try:
        while not stop_event.is_set():
            # Check for new positions from detection thread
            ball_pos, player_pos = plotter.poll_queue()
            if ball_pos is not None or player_pos is not None:
                plotter.update(ball_pos, player_pos)
            
            # Keep matplotlib responsive
            plt.pause(0.01)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        detection_thread.join(timeout=2.0)
        cv2.destroyAllWindows()

