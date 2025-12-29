import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from board_config import board_config_letter
from board_est import BoardEstimator
from ball_det import BallDetector
from cam_config import global_cam

matplotlib.use('TkAgg')


class BoardPlotter2D:
    """2D top-down view of board with ball position."""
    
    def __init__(self, board_config, update_interval=10):
        """Initialize plotter.
        
        Args:
            board_config: Board configuration
            update_interval: Update plot every N frames
        """
        self.config = board_config
        self.update_interval = update_interval
        self.frame_count = 0
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.fig.canvas.manager.set_window_title('Board 2D View')
        
        # Initialize board outline (will be drawn once)
        self._draw_board_outline()
        
        # Ball scatter plot (will be updated)
        self.ball_scatter = self.ax.scatter([], [], c='orange', s=200, marker='o', 
                                           edgecolors='black', linewidths=2, zorder=10)
        
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
    
    def update(self, ball_pos):
        """Update ball position.
        
        Args:
            ball_pos: (x, y) tuple in board coordinates, or None
        """
        self.frame_count += 1
        
        if self.frame_count % self.update_interval != 0:
            return
        
        if ball_pos is not None:
            x, y = ball_pos
            self.ball_scatter.set_offsets([[x, y]])
        else:
            # Hide ball by setting empty array with proper shape
            self.ball_scatter.set_offsets(np.empty((0, 2)))
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


if __name__ == "__main__":
    import cv2
    from cam_config import global_cam
    
    # Setup
    be = BoardEstimator(board_config_letter, K=global_cam.K, D=global_cam.D, rotate_180=True)
    ball_detector = BallDetector()
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
        if result is not None:
            board_T, pnp_result = result
            
            # Detect ball
            contour = ball_detector.detect(frame, drawing_frame=drawing_frame)
            if contour is not None:
                centroid = ball_detector.get_centroid(contour)
                if centroid is not None:
                    # Project to board coordinates
                    ball_pos = be.project_point_to_board(pnp_result, centroid, frame.shape)
                    
                    # Draw on frame
                    cx, cy = centroid
                    cv2.circle(drawing_frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(drawing_frame, f"Ball: ({ball_pos[0]:.3f}, {ball_pos[1]:.3f})m", 
                               (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Update plotter
        plotter.update(ball_pos)
        
        # Display frame
        cv2.imshow("Camera View", drawing_frame)
        cv2.setWindowProperty("Camera View", cv2.WND_PROP_TOPMOST, 1)
