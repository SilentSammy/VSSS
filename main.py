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
import time

def rotate_commands(x_ref, y_ref, angle):
    """Rotate commands from one frame to robot frame accounting for orientation.
    
    Args:
        x_ref: X command in reference frame
        y_ref: Y command in reference frame
        angle: Robot orientation angle in radians relative to reference frame
        
    Returns:
        tuple: (x, y) commands in robot frame
    """
    x = x_ref * np.cos(angle) - y_ref * np.sin(angle)
    y = x_ref * np.sin(angle) + y_ref * np.cos(angle)
    return x, y

def image_servo_demo():
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
            loop_start = time.perf_counter()
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
            # Get frame
            t0 = time.perf_counter()
            frame = global_cam.get_frame()
            drawing_frame = frame.copy()
            t_frame = time.perf_counter() - t0

            # Detect markers
            t0 = time.perf_counter()
            detections = ad.detect(frame, drawing_frame=drawing_frame)
            t_detect = time.perf_counter() - t0
            
            # Process largest detected ArUco (closest marker)
            t0 = time.perf_counter()
            if detections:
                marker = max(detections, key=lambda d: d.area)
                angle = marker.angle - np.radians(90)  # Adjust based on desired angle (angle is measured from x axis, so subtract 90° to get from y axis)
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
                x, y = rotate_commands(x_cam, y_cam, angle)
                
                # Create algorithmic command
                auto_cmd = {'x': x, 'y': y, 'w': w}
            else:
                # No marker detected
                auto_cmd = {'x': 0, 'y': 0, 'w': 0}
            t_control = time.perf_counter() - t0
            
            # Send command with manual override
            t0 = time.perf_counter()
            client.set_velocity(get_manual_override(auto_cmd))
            t_ble = time.perf_counter() - t0

            # Display
            t0 = time.perf_counter()
            cv2.imshow("Vision Sensor", drawing_frame)
            cv2.setWindowProperty("Vision Sensor", cv2.WND_PROP_TOPMOST, 1)
            t_display = time.perf_counter() - t0
            
            # Print timing info
            loop_time = time.perf_counter() - loop_start
            fps = 1.0 / loop_time if loop_time > 0 else 0
            print(f"FPS: {fps:5.1f} | Frame: {t_frame*1000:5.1f}ms | Detect: {t_detect*1000:5.1f}ms | Control: {t_control*1000:5.1f}ms | BLE: {t_ble*1000:5.1f}ms | Display: {t_display*1000:5.1f}ms")
    finally:
        # Stop all motors before disconnect
        client.stop()
        client.disconnect()

def board_servo_demo():
    from game_det import game_detector, GamePlotter2D
    from board_config import global_board_config
    
    # Real setup configuration
    plotter = GamePlotter2D(
        global_board_config,
        player_width=0.05,
        player_length=0.075
    )
    
    # PID controller for aiming at ball
    pid_w = PID(Kp=0.6, Ki=0.02, Kd=0.02, setpoint=0)
    pid_w.output_limits = (-0.5, 0.5)
    
    # Connect to mecanum car
    client = MecanumBLEClient(resolution=0.05)
    client.connect()
    
    try:
        while True:
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break
            
            frame = global_cam.get_frame()
            
            if frame is None:
                continue
            
            drawing_frame = frame.copy()
            
            # Detect game state
            game_state = game_detector.detect(frame, drawing_frame)
            
            # Update 2D plot
            plotter.update(game_state)
            
            # Annotate camera view with ball positions
            for i, ball in enumerate(game_state.balls):
                cv2.putText(drawing_frame, f"Ball: ({ball.x:.3f}, {ball.y:.3f})m",
                           (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Annotate camera view with player positions
            for i, player in enumerate(game_state.players):
                angle_deg = np.degrees(player.angle)
                cv2.putText(drawing_frame, f"Player {player.id}: ({player.x:.3f}, {player.y:.3f})m, {angle_deg:.1f}deg",
                           (10, 60 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 255), 2)
            
            # Calculate aiming angle if we have both player and ball
            auto_cmd = {'x': 0, 'y': 0, 'w': 0}
            if game_state.players and game_state.balls:
                player = game_state.players[0]
                ball = game_state.balls[0]
                
                # Calculate angle to ball
                dx = ball.x - player.x
                dy = ball.y - player.y
                target_angle = np.arctan2(dy, dx)
                
                # Calculate angle error
                angle_error = (target_angle - player.angle + np.pi) % (2 * np.pi) - np.pi
                
                # Apply deadband to prevent jittering when close
                if abs(angle_error) < np.radians(5):  # 5 degree deadband
                    w = 0
                else:
                    # PID control for rotation
                    w = pid_w(angle_error)
                auto_cmd['w'] = w
                
                # Debug output
                print(f"Player angle: {np.degrees(player.angle):6.1f}° | "
                      f"Ball pos: ({ball.x:6.3f}, {ball.y:6.3f}) | "
                      f"Target: {np.degrees(target_angle):6.1f}° | "
                      f"Error: {np.degrees(angle_error):6.1f}° | "
                      f"w: {w:5.2f}")
                
                # Annotate angle to ball
                cv2.putText(drawing_frame, f"Target angle: {np.degrees(target_angle):.1f} deg",
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(drawing_frame, f"Angle error: {np.degrees(angle_error):.1f} deg",
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Send manual control commands (manual can override x, y; w is auto)
            client.set_velocity(get_manual_override(auto_cmd))
            
            # Display camera view
            cv2.imshow("Game Detection", drawing_frame)
            cv2.setWindowProperty("Game Detection", cv2.WND_PROP_TOPMOST, 1)
    finally:
        client.stop()
        client.disconnect()
        plotter.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # image_servo_demo()
    board_servo_demo()
