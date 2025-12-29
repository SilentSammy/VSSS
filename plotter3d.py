import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use non-threaded backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class BoardPlotter3D:
    """Real-time 3D visualization of board pose relative to camera.
    
    Can display either:
    - Camera at origin, board moving (camera_at_origin=True, default)
    - Board at origin, camera moving (camera_at_origin=False)
    """
    
    def __init__(self, board_config, axis_limit=1.0, update_interval=10, camera_at_origin=True):
        """Initialize 3D plotter.
        
        Args:
            board_config: BoardConfig instance to get board dimensions
            axis_limit: Axis limits in meters (default: 1.0m cube)
            update_interval: Update plot every N frames (default: 10)
            camera_at_origin: If True, camera at origin and board moves.
                            If False, board at origin and camera moves (default: True)
        """
        self.board_config = board_config
        self.axis_limit = axis_limit
        self.update_interval = update_interval
        self.camera_at_origin = camera_at_origin
        self.frame_count = 0
        
        # Get board dimensions
        self.board_width, self.board_height = board_config.get_board_dimensions()
        
        # Setup plot
        plt.ion()
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Initialize artists (will be updated)
        self.board_poly = None
        self.board_quivers = []
        self.camera_artists = []
        
        self._setup_plot()
        
        # Draw fixed reference frame based on mode
        if self.camera_at_origin:
            self._draw_camera_frame()
        else:
            self._draw_board_frame()
        
        # Initialize moving object artists
        self.board_poly = None
        self.board_quivers = []
        
        # Show initially
        plt.show(block=False)
        plt.pause(0.001)
        
    def _setup_plot(self):
        """Configure 3D axes and labels."""
        self.ax.set_xlim([-self.axis_limit, self.axis_limit])
        self.ax.set_ylim([-self.axis_limit, self.axis_limit])
        self.ax.set_zlim([0, 2 * self.axis_limit])
        
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Board Pose Estimation')
        
        # Set viewing angle
        self.ax.view_init(elev=20, azim=45)
        
    def _draw_camera_frame(self):
        """Draw camera coordinate frame at origin (only once)."""
        axis_length = 0.2
        
        # Camera coordinate axes at origin
        self.camera_artists.append(
            self.ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', arrow_length_ratio=0.3, linewidth=2)
        )
        self.camera_artists.append(
            self.ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', arrow_length_ratio=0.3, linewidth=2)
        )
        self.camera_artists.append(
            self.ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', arrow_length_ratio=0.3, linewidth=2)
        )
    
    def _draw_board_frame(self):
        """Draw board coordinate frame and plane at origin (only once)."""
        axis_length = 0.2
        
        # Board coordinate axes at origin
        self.camera_artists.append(
            self.ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', arrow_length_ratio=0.3, linewidth=2)
        )
        self.camera_artists.append(
            self.ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', arrow_length_ratio=0.3, linewidth=2)
        )
        self.camera_artists.append(
            self.ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', arrow_length_ratio=0.3, linewidth=2)
        )
        
        # Draw board plane at origin
        corners = self._get_board_corners()
        verts = [corners]
        board_poly = Poly3DCollection(verts, alpha=0.3, facecolor='gray', edgecolor='black', linewidth=2)
        self.ax.add_collection3d(board_poly)
        self.camera_artists.append(board_poly)
        
    def _get_board_corners(self):
        """Get board corners in board's local frame.
        
        Returns:
            np.ndarray: 4x3 array of corner positions (centered at origin)
        """
        w, h = self.board_width, self.board_height
        
        # Corners centered at origin, in XY plane (Z=0)
        corners = np.array([
            [-w/2, -h/2, 0],  # Bottom-left
            [ w/2, -h/2, 0],  # Bottom-right
            [ w/2,  h/2, 0],  # Top-right
            [-w/2,  h/2, 0],  # Top-left
        ])
        
        return corners
    
    def _transform_points(self, points, T):
        """Transform points by homogeneous transformation matrix.
        
        Args:
            points: Nx3 array of points
            T: 4x4 transformation matrix
            
        Returns:
            Nx3 array of transformed points
        """
        # Convert to homogeneous coordinates
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])
        
        # Apply transformation
        points_transformed = (T @ points_h.T).T
        
        # Convert back to 3D
        return points_transformed[:, :3]
    
    def update(self, board_T):
        """Update visualization with new board pose.
        
        Args:
            board_T: 4x4 homogeneous transformation matrix from camera to board
        """
        # Only update every N frames to reduce lag
        self.frame_count += 1
        if self.frame_count % self.update_interval != 0:
            return
        
        # If board is at origin, invert the transform to show camera moving
        if not self.camera_at_origin:
            board_T = np.linalg.inv(board_T)
        
        # Remove old visualization
        if self.board_poly is not None:
            self.board_poly.remove()
        for quiver in self.board_quivers:
            quiver.remove()
        self.board_quivers.clear()
        
        # Draw based on mode
        if self.camera_at_origin:
            # Draw board plane and axes at transformed position
            self._draw_moving_board(board_T)
        else:
            # Draw camera axes only (no plane) at transformed position
            self._draw_moving_camera(board_T)
        
        # Refresh display (non-blocking)
        self.fig.canvas.flush_events()
    
    def _draw_moving_board(self, board_T):
        """Draw board plane and coordinate frame at given transform."""
        # Get board corners and transform to camera frame
        corners_local = self._get_board_corners()
        corners_camera = self._transform_points(corners_local, board_T)
        
        # Draw board as filled polygon
        verts = [corners_camera]
        self.board_poly = Poly3DCollection(verts, alpha=0.5, facecolor='cyan', edgecolor='darkblue', linewidth=2)
        self.ax.add_collection3d(self.board_poly)
        
        # Draw board coordinate frame
        board_origin = board_T[:3, 3]
        axis_length = 0.15
        
        # Extract rotation axes from transformation matrix
        x_axis = board_T[:3, 0] * axis_length
        y_axis = board_T[:3, 1] * axis_length
        z_axis = board_T[:3, 2] * axis_length
        
        # Draw axes
        self.board_quivers.append(
            self.ax.quiver(board_origin[0], board_origin[1], board_origin[2],
                          x_axis[0], x_axis[1], x_axis[2],
                          color='r', arrow_length_ratio=0.3, linewidth=1.5, alpha=0.7)
        )
        self.board_quivers.append(
            self.ax.quiver(board_origin[0], board_origin[1], board_origin[2],
                          y_axis[0], y_axis[1], y_axis[2],
                          color='g', arrow_length_ratio=0.3, linewidth=1.5, alpha=0.7)
        )
        self.board_quivers.append(
            self.ax.quiver(board_origin[0], board_origin[1], board_origin[2],
                          z_axis[0], z_axis[1], z_axis[2],
                          color='b', arrow_length_ratio=0.3, linewidth=1.5, alpha=0.7)
        )
    
    def _draw_moving_camera(self, cam_T):
        """Draw camera coordinate frame (axes only) at given transform."""
        camera_origin = cam_T[:3, 3]
        axis_length = 0.15
        
        # Extract rotation axes from transformation matrix
        x_axis = cam_T[:3, 0] * axis_length
        y_axis = cam_T[:3, 1] * axis_length
        z_axis = cam_T[:3, 2] * axis_length
        
        # Draw axes
        self.board_quivers.append(
            self.ax.quiver(camera_origin[0], camera_origin[1], camera_origin[2],
                          x_axis[0], x_axis[1], x_axis[2],
                          color='r', arrow_length_ratio=0.3, linewidth=1.5, alpha=0.7)
        )
        self.board_quivers.append(
            self.ax.quiver(camera_origin[0], camera_origin[1], camera_origin[2],
                          y_axis[0], y_axis[1], y_axis[2],
                          color='g', arrow_length_ratio=0.3, linewidth=1.5, alpha=0.7)
        )
        self.board_quivers.append(
            self.ax.quiver(camera_origin[0], camera_origin[1], camera_origin[2],
                          z_axis[0], z_axis[1], z_axis[2],
                          color='b', arrow_length_ratio=0.3, linewidth=1.5, alpha=0.7)
        )
    
    def close(self):
        """Close the plot window."""
        plt.close(self.fig)

if __name__ == "__main__":
    import cv2
    from board_est import BoardEstimator
    from board_config import board_config
    from cam_config import global_cam
    
    be = BoardEstimator(
        board_config=board_config,
        K=global_cam.K,
        D=global_cam.D,
        # rotate_180=False
    )
    
    plotter = BoardPlotter3D(
        board_config,
        axis_limit=0.5,
        # camera_at_origin=True,
        camera_at_origin=False,
    )
    
    while True:
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
        # Get frame
        frame = global_cam.get_frame()
        drawing_frame = frame.copy()
    
        # Estimate
        res = be.get_board_transform(frame, drawing_frame=drawing_frame)
    
        if res is not None:
            board_T, _ = res
            plotter.update(board_T)
    
        # Display
        cv2.imshow("Camera", drawing_frame)
        cv2.setWindowProperty("Camera", cv2.WND_PROP_TOPMOST, 1)
