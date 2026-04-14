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
        
        corners, ids, _ = detector.detectMarkers(frame)
        
        w = 0.0
        
        if ids is not None and len(ids) > 0:
            marker_corners = corners[0][0]
            
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            vec = marker_corners[0] - marker_corners[2]
            angle = -np.arctan2(vec[1], vec[0]) - np.pi / 4
            angle = (angle + np.pi) % (2 * np.pi) - np.pi
            
            error = angle - np.pi / 2
            w = -error * 0.4
            
            print(f"Angle: {np.degrees(angle):6.1f}° | Error: {np.degrees(error):6.1f}° | w: {w:5.2f}")
        
        if inp.is_pressed('q'):
            break
        
        velocity = {'x': 0, 'y': 0, 'w': w}
        car.set_velocity(velocity)
        
        cv2.imshow("ArUco Demo", frame)
        time.sleep(0.02)
finally:
    car.stop()
    car.disconnect()
    cap.release()
    cv2.destroyAllWindows()
