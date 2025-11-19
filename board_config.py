import cv2
import numpy as np

class GridboardConfig:
    """Configuration for ArUco GridBoard with automatic marker separation calculation."""
    def __init__(self, dictionary, size, marker_length, board_width):
        """Initialize GridBoard configuration with automatic marker separation."""
        self.dictionary = dictionary
        self.size = size
        self.marker_length = marker_length
        self.board_width = board_width
        
        # Calculate marker separation automatically
        # board_width = (n_markers * marker_length) + ((n_markers - 1) * separation)
        # separation = (board_width - n_markers * marker_length) / (n_markers - 1)
        cols = size[0]
        total_marker_width = cols * marker_length
        if cols > 1:
            self.marker_separation = (board_width - total_marker_width) / (cols - 1)
        else:
            self.marker_separation = 0.0
        
        # Create the board and detector
        self.board = cv2.aruco.GridBoard(
            size=self.size,
            markerLength=self.marker_length,
            markerSeparation=self.marker_separation,
            dictionary=self.dictionary
        )
        self.detector = cv2.aruco.ArucoDetector(self.dictionary)
        
        # Calculate board center
        cols, rows = self.size
        total_x = (cols * self.marker_length) + ((cols - 1) * self.marker_separation)
        total_y = (rows * self.marker_length) + ((rows - 1) * self.marker_separation)
        self.center = np.array([total_x / 2.0, total_y / 2.0, 0.0], dtype=np.float64)
    
    def detect_corners(self, frame, drawing_frame=None):
        """Detect markers in a frame and return corners and IDs.
        
        Args:
            frame: Input image/frame to detect markers in
            drawing_frame: Optional frame to draw detected markers on
            
        Returns:
            tuple: (corners, ids) where corners is a list of detected marker corners
                   and ids is an array of corresponding marker IDs
        """
        corners, ids, _ = self.detector.detectMarkers(frame)
        
        if drawing_frame is not None and ids is not None:
            cv2.aruco.drawDetectedMarkers(drawing_frame, corners, ids)
        
        return corners, ids
    
    def generate_image(self, width_px=1200, margin_px=0):
        """Generate board image in pixels."""
        # Calculate actual board dimensions
        cols, rows = self.size
        actual_width = cols * self.marker_length + (cols - 1) * self.marker_separation
        actual_height = rows * self.marker_length + (rows - 1) * self.marker_separation
        
        # Calculate aspect ratio and image dimensions
        aspect_ratio = actual_height / actual_width
        height_px = int(width_px * aspect_ratio)
        
        # Generate board image
        img = self.board.generateImage((width_px, height_px), marginSize=margin_px, borderBits=1)
        
        return img
    
    def save_image(self, filepath, width_px=1200, margin_px=0):
        """Save board image to file."""
        img = self.generate_image(width_px=width_px, margin_px=margin_px)
        cv2.imwrite(filepath, img)
        return filepath

# Instantiate board configuration
board_config = GridboardConfig(
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    size=(3, 4),
    marker_length=0.075,
    board_width=0.60
)

if __name__ == "__main__":
    # Example usage: save board image
    board_config.save_image("gridboard.png", width_px=1200)
