import cv2
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from board_est import BoardEstimator
from board_config import board_config

def get_image(vision_sensor_handle):
    sim.handleVisionSensor(vision_sensor_handle)
    img, resolution = sim.getVisionSensorImg(vision_sensor_handle)
    img = np.frombuffer(img, dtype=np.uint8).reshape((resolution[1], resolution[0], 3))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.flip(img, 0)
    return img

client = RemoteAPIClient('localhost', 23000)
sim = client.getObject('sim')
cam_handle = sim.getObject(f"/visionSensor")

de = BoardEstimator(
    board_config=board_config,
    K=np.array([[444,   0, 256], [  0, 444, 256], [  0,   0,   1]], dtype=np.float32),
    D=np.zeros(5)
)

while True:
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # Get frame
    frame = get_image(cam_handle)
    drawing_frame = frame.copy()

    # Estimate
    res = de.get_board_transform(frame, drawing_frame=drawing_frame)

    if res is not None:
        cam_T, _ = res

    # Display
    cv2.imshow("Vision Sensor", drawing_frame)
    cv2.setWindowProperty("Vision Sensor", cv2.WND_PROP_TOPMOST, 1)
