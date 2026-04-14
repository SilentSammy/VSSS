import time
import cv2
import numpy as np
import user_input as inp
from mecanum_client import MecanumBLEClient

DEVICE_NAME = "Therian00"
CAMERA_INDEX = 1

cap = cv2.VideoCapture(CAMERA_INDEX)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
detector = cv2.aruco.ArucoDetector(dictionary)

car = MecanumBLEClient(device_name=DEVICE_NAME)
car.connect()

try:
    while True:
        if cv2.waitKey(1) & 0xFF == 27:
            break
        
        ret, frame = cap.read()
        if not ret:
            continue
        
        height, width = frame.shape[:2]
        center_x = width // 2
        center_y = height // 2
        
        corners, ids, _ = detector.detectMarkers(frame)
        
        x = 0.0
        y = 0.0
        w = 0.0
        
        if ids is not None and len(ids) > 0:
            marker_corners = corners[0][0]
            marker_center_x = int(np.mean(marker_corners[:, 0]))
            marker_center_y = int(np.mean(marker_corners[:, 1]))
            
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.circle(frame, (marker_center_x, marker_center_y), 5, (0, 255, 0), -1)
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
            cv2.line(frame, (center_x, center_y), (marker_center_x, marker_center_y), (0, 255, 255), 2)
            
            error_x = (marker_center_x - center_x) / width
            error_y = (marker_center_y - center_y) / height
            
            vec = marker_corners[0] - marker_corners[2]
            angle = np.arctan2(vec[1], vec[0]) - np.pi / 2
            
            y = -error_x * 0.5
            x = error_y * 0.5
            w = -angle * 0.4
        
        if inp.is_pressed('q'):
            break
        
        velocity = {'x': x, 'y': y, 'w': w}
        car.set_velocity(velocity)
        
        cv2.imshow("ArUco Demo", frame)
        time.sleep(0.02)
finally:
    car.stop()
    car.disconnect()
    cap.release()
    cv2.destroyAllWindows()
