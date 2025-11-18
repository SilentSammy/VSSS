import cv2
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from cam_est import CamEstimator

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

de = CamEstimator(
    K = np.array([[444,   0, 256], [  0, 444, 256], [  0,   0,   1]], dtype=np.float32),
    D = np.zeros(5),  # [0, 0, 0, 0, 0]
    # board = cv2.aruco.CharucoBoard(
    #     size=(9, 24),
    #     squareLength=0.1,
    #     markerLength=0.08,
    #     dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    # )
    board = cv2.aruco.GridBoard(
        size=(3, 4),
        markerLength=0.075,
        markerSeparation=(0.60 - 3 * 0.075) / 2,
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    )
)

while True:
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # Get frame
    frame = get_image(cam_handle)
    drawing_frame = frame.copy()

    # Estimate
    res = de.get_camera_transform(frame, drawing_frame=drawing_frame)

    if res is not None:
        cam_T, _ = res

    # Display
    cv2.imshow("Vision Sensor", drawing_frame)
    cv2.setWindowProperty("Vision Sensor", cv2.WND_PROP_TOPMOST, 1)


