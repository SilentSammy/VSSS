import cv2
import numpy as np

class Camera:
    """Camera configuration with intrinsics and frame acquisition."""
    
    def __init__(self, K, D, frame_getter):
        """Initialize camera.
        
        Args:
            K: Camera intrinsic matrix
            D: Distortion coefficients
            frame_getter: Callable that returns a frame
        """
        self.K = K
        self.D = D
        self.frame_getter = frame_getter
    
    def get_frame(self):
        """Get frame from frame_getter."""
        return self.frame_getter()

# Camera frame getters
def _get_sim_image():
    _get_sim_image.cam_handle = getattr(_get_sim_image, 'cam_handle', None)
    if _get_sim_image.cam_handle is None:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        client = RemoteAPIClient('localhost', 23000)
        sim = client.getObject('sim')
        # _get_sim_image.cam_handle = sim.getObjectHandle('/visionSensor[1]')
        _get_sim_image.cam_handle = sim.getObjectHandle('/visionSensor[0]')
        _get_sim_image.sim = sim

    sim = _get_sim_image.sim
    vision_sensor_handle = _get_sim_image.cam_handle

    sim.handleVisionSensor(vision_sensor_handle)
    img, resolution = sim.getVisionSensorImg(vision_sensor_handle)
    img = np.frombuffer(img, dtype=np.uint8).reshape((resolution[1], resolution[0], 3))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.flip(img, 0)
    return img

def _get_droidcam_image():
    _get_droidcam_image.cap = getattr(_get_droidcam_image, 'cap', None)
    if _get_droidcam_image.cap is None:
        _get_droidcam_image.cap = cv2.VideoCapture("http://192.168.1.211:4747/video")
    
    ret, frame = _get_droidcam_image.cap.read()
    if not ret:
        return None
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame

sim_cam = Camera(
    K=np.array([[444,   0, 256], [  0, 444, 256], [  0,   0,   1]], dtype=np.float32),
    D=np.zeros(5),
    frame_getter=_get_sim_image
)

droidcam = Camera(
    K=np.array([[487.14566155, 0., 321.7888109], [0., 487.60075097, 239.38896134], [0., 0., 1.]], dtype=np.float32),
    # D=np.array([0.33819757, 1.36709606, -6.17042008, 8.65929659], dtype=np.float32),
    D=np.zeros(5),
    frame_getter=_get_droidcam_image
)

global_cam = droidcam
global_cam = sim_cam
