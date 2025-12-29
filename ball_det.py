import cv2
import numpy as np


class BallDetector:
    """Detects ball in image using HSV thresholding."""
    
    def __init__(self, hsv_lower=None, hsv_upper=None):
        """Initialize detector.
        
        Args:
            hsv_lower: Lower HSV threshold (H, S, V). If None, loads from file or uses defaults.
            hsv_upper: Upper HSV threshold (H, S, V). If None, loads from file or uses defaults.
        """
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
    
    def detect(self, frame, drawing_frame=None):
        """Detect ball in frame.
        
        Args:
            frame: BGR image
            drawing_frame: Optional frame to draw ball outline on
            
        Returns:
            Ball contour or None if not found
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
            return None
        
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
        
        # Draw on frame if provided
        if drawing_frame is not None and best_contour is not None:
            cv2.drawContours(drawing_frame, [best_contour], -1, (0, 255, 0), 2)
        
        return best_contour
    
    @staticmethod
    def get_centroid(contour):
        """Get centroid of contour.
        
        Args:
            contour: OpenCV contour
            
        Returns:
            (x, y) tuple or None if contour is invalid
        """
        if contour is None:
            return None
        
        M = cv2.moments(contour)
        if M['m00'] == 0:
            return None
        
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy)

if __name__ == "__main__":
    from cam_config import global_cam
    
    detector = BallDetector()
    
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
            detector.hsv_lower = np.array(lower, dtype=np.uint8)
            detector.hsv_upper = np.array(upper, dtype=np.uint8)
            
            # Save to file
            with open('ball_thresholds.txt', 'w') as f:
                f.write(f"{lower[0]},{lower[1]},{lower[2]}\n")
                f.write(f"{upper[0]},{upper[1]},{upper[2]}\n")
            
            print(f"Updated thresholds: Lower={lower}, Upper={upper}")
    
    cv2.namedWindow("Ball Detection")
    cv2.setMouseCallback("Ball Detection", mouse_callback)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):  # Reset to default yellow thresholds
            detector.hsv_lower = np.array([20, 100, 100], dtype=np.uint8)
            detector.hsv_upper = np.array([30, 255, 255], dtype=np.uint8)
            with open('ball_thresholds.txt', 'w') as f:
                f.write("20,100,100\n")
                f.write("30,255,255\n")
            print("Reset to default yellow thresholds: Lower=(20,100,100), Upper=(30,255,255)")
        
        # Get frame
        frame = global_cam.get_frame()
        drawing_frame = frame.copy()
        click_state['frame'] = frame
        
        # Detect ball
        contour = detector.detect(frame, drawing_frame=drawing_frame)
        
        # Get and display centroid
        if contour is not None:
            centroid = detector.get_centroid(contour)
            if centroid is not None:
                cx, cy = centroid
                cv2.circle(drawing_frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(drawing_frame, f"Ball: ({cx}, {cy})", (cx + 10, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Display instructions
        cv2.putText(drawing_frame, "Double-click ball to configure | R to reset", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Display
        cv2.imshow("Ball Detection", drawing_frame)
        cv2.setWindowProperty("Ball Detection", cv2.WND_PROP_TOPMOST, 1)
