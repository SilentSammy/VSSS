import cv2
import numpy as np
import time
from pathlib import Path
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
            norm_centroid = (norm_centroid_x, -norm_centroid_y)
            
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
            with Path(__file__).with_name('ball_thresholds.txt').open() as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    lower = tuple(map(int, lines[0].strip().split(',')))
                    upper = tuple(map(int, lines[1].strip().split(',')))
                    return lower, upper
        except (FileNotFoundError, ValueError, IndexError):
            pass
        
        # Default to blue if no saved calibration is available.
        return (95, 50, 30), (130, 255, 255)
    
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

