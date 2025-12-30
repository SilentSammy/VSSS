class ObjectLocalizer:
    """Locates detected objects in 3D board coordinates with parallax correction."""
    
    def __init__(self, detector, board_estimator, height=0.0):
        """Initialize localizer with detector, board estimator, and object height.
        
        Args:
            detector: ObjectDetector instance for 2D detection
            board_estimator: BoardEstimator for projection to board coordinates
            height: Object height above board (m) for parallax correction
        """
        self.detector = detector
        self.board_estimator = board_estimator
        self.height = height
    
    def localize(self, frame, pnp_result, drawing_frame=None):
        """Detect object and return 3D board coordinates with parallax correction.
        
        Args:
            frame: Input image frame
            pnp_result: PnP result from board pose estimation
            drawing_frame: Optional frame to draw detections on
            
        Returns:
            (x, y, z) tuple in board coordinates (meters), or None if not detected
        """
        centroid, contour = self.detector.detect(frame, drawing_frame)
        
        if centroid is None:
            return None
        
        x, y = self.board_estimator.project_point_to_board(
            pnp_result, centroid, frame.shape, z=self.height
        )
        
        return (x, y, self.height)
