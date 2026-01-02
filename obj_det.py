import cv2
import numpy as np
import time
from dataclasses import dataclass

@dataclass
class DetectedObject:
    contour : np.ndarray = None
    centroid : tuple = None  # (x, y)
    norm_centroid : tuple = None  # normalized (x, y)
    area : float = None  # contour area in pixels
    timestamp : float = None  # time.time() when object was detected
    parent : 'ObjectDetector' = None
    source_image = None  # The original image this object was detected in

@dataclass
class DetectedAruco(DetectedObject):
    id : int = None  # ArUco marker ID
    dict : cv2.aruco.Dictionary = None  # ArUco dictionary used
    angle : float = None  # Marker orientation in radians

class ObjectDetector:
    """Base class for object detection."""
    
    object_class = DetectedObject
    
    def _detect(self, frame, drawing_frame=None):
        """Detect objects and return list of DetectedObject instances.
        
        Subclasses should populate at minimum the contour field and any
        custom fields for each detection, then return the list. The base
        detect() method will fill in all generic fields.
        
        Returns:
            List of DetectedObject instances (or empty list if none found)
        """
        pass
    
    def detect(self, frame, drawing_frame=None):
        """Detect objects and return list of DetectedObject instances."""
        detections = self._detect(frame, drawing_frame)
        
        if detections is None or not detections:
            return []
        
        # Get image dimensions for normalization
        height, width = frame.shape[:2]
        
        results = []
        for detection in detections:
            # Get contour (subclass should have populated this)
            contour = detection.contour
            if contour is None or len(contour) == 0:
                continue
            
            # Compute centroid
            centroid = self.get_centroid(contour)
            if centroid is None:
                continue
            
            # Normalize centroid
            norm_centroid_x = (centroid[0] - width / 2) / (width / 2)
            norm_centroid_y = (centroid[1] - height / 2) / (height / 2)
            norm_centroid = (norm_centroid_x, norm_centroid_y)
            
            # Compute area
            area = cv2.contourArea(contour)
            
            # Populate generic fields
            detection.centroid = centroid
            detection.norm_centroid = norm_centroid
            detection.area = area
            detection.timestamp = time.time()
            detection.parent = self
            detection.source_image = frame
            
            results.append(detection)
        
        return results
    
    @staticmethod
    def get_centroid(contour):
        """Get centroid (x, y) of contour or None if invalid."""
        if contour is None or len(contour) == 0:
            return None
        
        M = cv2.moments(contour)
        if M['m00'] == 0:
            return None
        
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy)

class BallDetector(ObjectDetector):
    """Detects ball in image using HSV thresholding."""
    
    def __init__(self, hsv_lower=None, hsv_upper=None):
        """Initialize detector with HSV thresholds (loads from file if None)."""
        if hsv_lower is None or hsv_upper is None:
            hsv_lower, hsv_upper = self._load_thresholds()
        
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
    
    def _load_thresholds(self):
        """Load thresholds from file or return defaults."""
        try:
            with open('ball_thresholds.txt', 'r') as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    lower = tuple(map(int, lines[0].strip().split(',')))
                    upper = tuple(map(int, lines[1].strip().split(',')))
                    return lower, upper
        except (FileNotFoundError, ValueError, IndexError):
            pass
        
        # Default to yellow ball
        return (20, 100, 100), (30, 255, 255)
    
    def _detect(self, frame, drawing_frame=None):
        """Detect ball contour using HSV thresholding.
        
        Args:
            frame: BGR image
            drawing_frame: Optional frame to draw ball outline on
            
        Returns:
            List containing at most one DetectedObject with contour populated
        """
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        
        # Convert to HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Threshold
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        
        # Morphological operations to remove noise
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Find most circular contour
        best_contour = None
        best_circularity = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Minimum area threshold
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            # Circularity: 4π*area / perimeter^2 (1.0 = perfect circle)
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            
            # Check minimum radius
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            if radius < 10:  # Minimum radius threshold
                continue
            
            if circularity > best_circularity:
                best_circularity = circularity
                best_contour = contour
        
        if best_contour is None:
            return []
        
        # Draw on frame if provided
        if drawing_frame is not None:
            cv2.drawContours(drawing_frame, [best_contour], -1, (0, 255, 0), 2)
        
        return [DetectedObject(contour=best_contour)]

class ArucoDetector(ObjectDetector):
    """Detects all ArUco markers in image."""
    
    object_class = DetectedAruco
    
    def __init__(self, aruco_dict):
        """Initialize detector with ArUco dictionary."""
        self.aruco_dict = aruco_dict
        self.detector = cv2.aruco.ArucoDetector(aruco_dict)
    
    @staticmethod
    def get_angle(contour):
        """Calculate ArUco marker orientation from contour.
        
        Args:
            contour: ArUco marker contour in format returned by _detect
            
        Returns:
            Angle in radians from top-left to top-right corner
        """
        if contour is None:
            return None
        
        # Reshape back to corner format: [[top-left, top-right, bottom-right, bottom-left]]
        corners = contour.reshape(1, 4, 2)
        top_left = corners[0][0]
        top_right = corners[0][1]
        
        # Vector from top-left to top-right gives marker orientation
        dx = top_right[0] - top_left[0]
        dy = top_right[1] - top_left[1]
        angle = np.arctan2(dy, dx)
        
        # Apply π/2 offset and negation
        return -(angle - np.pi/2)
    
    def _detect(self, frame, drawing_frame=None):
        """Detect all ArUco markers and return list of DetectedAruco instances.
        
        Args:
            frame: BGR image
            drawing_frame: Optional frame to draw marker outlines on
            
        Returns:
            List of DetectedAruco instances with contour, id, dict, and angle populated
        """
        corners, ids, _ = self.detector.detectMarkers(frame)
        
        if ids is None:
            return []
        
        detections = []
        for i, marker_id in enumerate(ids.flatten()):
            # Convert corner points to contour format
            contour = corners[i].reshape(-1, 1, 2).astype(np.int32)
            
            # Calculate angle
            angle = self.get_angle(contour)
            
            # Draw on frame if provided
            if drawing_frame is not None:
                cv2.drawContours(drawing_frame, [contour], -1, (0, 255, 0), 2)
            
            detections.append(DetectedAruco(
                contour=contour,
                id=int(marker_id),
                dict=self.aruco_dict,
                angle=angle
            ))
        
        return detections


if __name__ == "__main__":
    from cam_config import global_cam
    
    ball_detector = BallDetector()
    
    # State for mouse callback
    click_state = {'frame': None}
    
    def mouse_callback(event, x, y, flags, param):
        """Double-click to sample ball color and update thresholds."""
        if event == cv2.EVENT_LBUTTONDBLCLK and click_state['frame'] is not None:
            frame = click_state['frame']
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Sample 5x5 region around click to find brightest pixel
            region_size = 5
            y1 = max(0, y - region_size // 2)
            y2 = min(hsv.shape[0], y + region_size // 2 + 1)
            x1 = max(0, x - region_size // 2)
            x2 = min(hsv.shape[1], x + region_size // 2 + 1)
            
            region = hsv[y1:y2, x1:x2]
            
            # Find pixel with maximum V (brightness) in the region
            brightest_idx = np.unravel_index(region[:, :, 2].argmax(), region[:, :, 2].shape)
            h, s, v = region[brightest_idx]
            
            # Create thresholds with tolerance (convert to int to avoid uint8 overflow)
            # Use wider tolerance for S and V to capture darker/shadowed regions
            h_tol, s_tol, v_tol = 10, 50, 100
            lower = (max(0, int(h) - h_tol), max(0, int(s) - s_tol), max(0, int(v) - v_tol))
            upper = (min(179, int(h) + h_tol), min(255, int(s) + s_tol), min(255, int(v) + v_tol))
            
            # Update detector
            ball_detector.hsv_lower = np.array(lower, dtype=np.uint8)
            ball_detector.hsv_upper = np.array(upper, dtype=np.uint8)
            
            # Save to file
            with open('ball_thresholds.txt', 'w') as f:
                f.write(f"{lower[0]},{lower[1]},{lower[2]}\n")
                f.write(f"{upper[0]},{upper[1]},{upper[2]}\n")
            
            print(f"Updated thresholds: Lower={lower}, Upper={upper}")
    
    cv2.namedWindow("Object Detection")
    cv2.setMouseCallback("Object Detection", mouse_callback)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):  # Reset to default yellow thresholds
            ball_detector.hsv_lower = np.array([20, 100, 100], dtype=np.uint8)
            ball_detector.hsv_upper = np.array([30, 255, 255], dtype=np.uint8)
            with open('ball_thresholds.txt', 'w') as f:
                f.write("20,100,100\n")
                f.write("30,255,255\n")
            print("Reset to default yellow thresholds: Lower=(20,100,100), Upper=(30,255,255)")
        
        # Get frame
        frame = global_cam.get_frame()
        drawing_frame = frame.copy()
        click_state['frame'] = frame
        
        # Detect ball (new API returns list of DetectedObject)
        ball_detections = ball_detector.detect(frame, drawing_frame=drawing_frame)
        if ball_detections:
            ball = ball_detections[0]
            cv2.circle(drawing_frame, ball.centroid, 5, (0, 0, 255), -1)
            # Debug: print detection info
            print(f"Ball: centroid={ball.centroid}, norm_centroid=({ball.norm_centroid[0]:.3f}, {ball.norm_centroid[1]:.3f}), area={ball.area:.1f}")
        
        # Display instructions
        cv2.putText(drawing_frame, "Double-click ball to configure | R to reset | ESC to exit", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Display
        cv2.imshow("Object Detection", drawing_frame)
        cv2.setWindowProperty("Object Detection", cv2.WND_PROP_TOPMOST, 1)
