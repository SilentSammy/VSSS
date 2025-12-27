import cv2
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from board_est import BoardEstimator, get_cam_T
from board_config import board_config
from plotter3d import BoardPlotter3D

def get_image():
    get_image.cam_handle = getattr(get_image, 'cam_handle', None)
    if get_image.cam_handle is None:
        client = RemoteAPIClient('localhost', 23000)
        sim = client.getObject('sim')
        get_image.cam_handle = sim.getObjectHandle('/visionSensor[1]')
        get_image.sim = sim

    sim = get_image.sim
    vision_sensor_handle = get_image.cam_handle

    sim.handleVisionSensor(vision_sensor_handle)
    img, resolution = sim.getVisionSensorImg(vision_sensor_handle)
    img = np.frombuffer(img, dtype=np.uint8).reshape((resolution[1], resolution[0], 3))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.flip(img, 0)
    return img

be = BoardEstimator(
    board_config=board_config,
    K=np.array([[444,   0, 256], [  0, 444, 256], [  0,   0,   1]], dtype=np.float32),
    D=np.zeros(5)
)

plotter = BoardPlotter3D(board_config, axis_limit=0.5, camera_at_origin=True)

while True:
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # Get frame
    frame = get_image()
    drawing_frame = frame.copy()

    # Estimate
    res = be.get_board_transform(frame, drawing_frame=drawing_frame)

    if res is not None:
        board_T, _ = res
        plotter.update(board_T)

    # Display
    cv2.imshow("Vision Sensor", drawing_frame)
    cv2.setWindowProperty("Vision Sensor", cv2.WND_PROP_TOPMOST, 1)
