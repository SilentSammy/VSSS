import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rc'))
from mecanum_client import MecanumBLEClient, get_manual_override
import cv2
import numpy as np
from cv2 import aruco
from cam_config import global_cam
from obj_det import ArucoDetector
from simple_pid import PID

def main():
    ad = ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100))
    
    # Initialize PID controllers - gentle tuning for latency
    pid_w = PID(Kp=0.4, Ki=0, Kd=0.04, setpoint=0)
    pid_w.output_limits = (-0.5, 0.5)
    
    pid_x = PID(Kp=0.6, Ki=0, Kd=0.03, setpoint=0)
    pid_x.output_limits = (-0.5, 0.5)
    
    pid_y = PID(Kp=0.6, Ki=0, Kd=0.03, setpoint=0)
    pid_y.output_limits = (-0.5, 0.5)
    
    # Connect to mecanum car (resolution=0.05 rounds commands to reduce BLE writes)
    client = MecanumBLEClient(resolution=0.05)
    client.connect()
    
    try:
        while True:
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
            # Get frame
            frame = global_cam.get_frame()
            drawing_frame = frame.copy()

            # Detect markers
            detections = ad.detect(frame, drawing_frame=drawing_frame)
            
            # Process largest detected ArUco (closest marker)
            if detections:
                marker = max(detections, key=lambda d: d.area)
                angle = marker.angle - np.radians(90)  # Adjust based on marker orientation
                norm_x, norm_y = marker.norm_centroid
                
                # Annotate on drawing frame
                cv2.putText(drawing_frame, f"Angle: {np.degrees(angle):.1f} deg", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(drawing_frame, f"Centroid: ({norm_x:.2f}, {norm_y:.2f})", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw line from image center to marker centroid
                height, width = frame.shape[:2]
                center = (width // 2, height // 2)
                cv2.line(drawing_frame, center, marker.centroid, (0, 255, 255), 2)
                cv2.circle(drawing_frame, center, 5, (255, 0, 0), -1)  # Blue dot at center
                cv2.circle(drawing_frame, marker.centroid, 5, (0, 255, 0), -1)  # Green dot at centroid
                
                # Calculate PID control outputs
                w = -pid_w(angle)
                x_cam = pid_x(norm_y)  # Camera y (up/down) → Camera x
                y_cam = pid_y(norm_x)  # Camera x (left/right) → Camera y
                
                # Rotate translation commands to account for marker orientation
                # When marker is rotated, we need to rotate our control commands too
                x = x_cam * np.cos(angle) - y_cam * np.sin(angle)
                y = x_cam * np.sin(angle) + y_cam * np.cos(angle)
                
                # Create algorithmic command
                auto_cmd = {'x': x, 'y': y, 'w': w}
            else:
                # No marker detected
                auto_cmd = {'x': 0, 'y': 0, 'w': 0}
            
            # Send command with manual override
            client.set_velocity(get_manual_override(auto_cmd))

            # Display
            cv2.imshow("Vision Sensor", drawing_frame)
            cv2.setWindowProperty("Vision Sensor", cv2.WND_PROP_TOPMOST, 1)
    finally:
        # Stop all motors before disconnect
        client.stop()
        client.disconnect()

if __name__ == "__main__":
    main()
