import cv2
import numpy as np

def rotate_intrinsics(rotation, K, image_size):
    """Rotate camera intrinsics for rotated image.
    
    Args:
        rotation: cv2.ROTATE_* constant
        K: 3x3 intrinsic matrix
        image_size: (height, width) of original image before rotation
        
    Returns:
        K_rotated: Rotated intrinsic matrix
    """
    h, w = image_size
    K_rot = K.copy()
    
    if rotation == cv2.ROTATE_90_CLOCKWISE:
        # (h, w) -> (w, h)
        # (x, y) -> (h - y, x)
        K_rot[0, 2] = h - K[1, 2]  # new_cx = h - old_cy
        K_rot[1, 2] = K[0, 2]       # new_cy = old_cx
        
    elif rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
        # (h, w) -> (w, h)
        # (x, y) -> (y, w - x)
        K_rot[0, 2] = K[1, 2]       # new_cx = old_cy
        K_rot[1, 2] = w - K[0, 2]  # new_cy = w - old_cx
        
    elif rotation == cv2.ROTATE_180:
        # (h, w) -> (h, w)
        # (x, y) -> (w - x, h - y)
        K_rot[0, 2] = w - K[0, 2]  # new_cx = w - old_cx
        K_rot[1, 2] = h - K[1, 2]  # new_cy = h - old_cy
    
    return K_rot

# TODO: for cams such as droidcam, where we might need to rotate the frame, we should specify the unrotated intrinsics, then if user requires rotation, we can compute the new intrinsics accordingly.
class Camera:
    """Camera configuration with intrinsics and frame acquisition."""
    
    def __init__(self, K, D, frame_getter, rotation=None, image_shape_hw=None):
        """Initialize camera.
        
        Args:
            K: Camera intrinsic matrix
            D: Distortion coefficients
            frame_getter: Callable that returns a frame
            rotation: Optional cv2.ROTATE_* constant to apply to frames
            image_shape_hw: Optional (height, width) tuple. Required if rotation is specified.
        """
        # Validate that rotation and image_shape_hw are provided together
        if (rotation is None) != (image_shape_hw is None):
            raise ValueError("rotation and image_shape_hw must both be provided or both be None")
        
        self.frame_getter = frame_getter
        self.rotation = rotation
        self.D = D  # Distortion coefficients don't change (radially symmetric)
        
        # Compute rotated intrinsics if rotation specified
        if rotation is not None:
            self.K = rotate_intrinsics(rotation, K, image_shape_hw)
        else:
            self.K = K
    
    def get_frame(self):
        """Get frame from frame_getter."""
        frame = self.frame_getter()
        if frame is not None and self.rotation is not None:
            frame = cv2.rotate(frame, self.rotation)
        return frame

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

def _get_droidcam_image(rotation = None):
    ip = "http://10.173.208.191:4747/video"
    ip = "http://192.168.43.1:4747/video"
    ip = "http://192.168.1.211:4747/video"
    ip = "http://192.168.4.25:4747/video"
    ip = "http://192.168.137.210:4747/video"
    _get_droidcam_image.cap = getattr(_get_droidcam_image, 'cap', None)
    if _get_droidcam_image.cap is None:
        _get_droidcam_image.cap = cv2.VideoCapture(ip)
    
    ret, frame = _get_droidcam_image.cap.read()
    if not ret:
        return None
    if rotation is not None:
        frame = cv2.rotate(frame, rotation)
    return frame

def _get_usb_image():
    _get_usb_image.cap = getattr(_get_usb_image, 'cap', None)
    if _get_usb_image.cap is None:
        _get_usb_image.cap = cv2.VideoCapture(1)
    
    ret, frame = _get_usb_image.cap.read()
    if not ret:
        return None
    return frame

# Camera configurations
sim_cam = Camera(
    K=np.array([[444,   0, 256], [  0, 444, 256], [  0,   0,   1]], dtype=np.float32),
    D=np.zeros(5),
    frame_getter=_get_sim_image
)

droidcam = Camera(
    K=np.array([[476.21413568, 0., 324.64535892], [0., 476.57490297, 242.01755433], [0., 0., 1.]], dtype=np.float32),
    # D=np.array([0.37628059, 0.8828322, -4.22102342, 5.72132593], dtype=np.float32),
    D=np.zeros(5),
    frame_getter=_get_droidcam_image,
    rotation=cv2.ROTATE_90_CLOCKWISE,
    image_shape_hw=(480, 640)  # height, width before rotation
)

usb_cam = Camera(
    K=np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32),
    D=np.zeros(5),
    frame_getter=_get_usb_image
)

# Last assignment gets used as global_cam
global_cam = usb_cam
global_cam = sim_cam
global_cam = droidcam

if __name__ == "__main__":
    while True:
        if cv2.waitKey(1) & 0xFF == 27:
            break
        
        # Get frame
        frame = global_cam.get_frame()
        drawing_frame = frame.copy()

        # Display
        cv2.imshow("Vision Sensor", drawing_frame)
        cv2.setWindowProperty("Vision Sensor", cv2.WND_PROP_TOPMOST, 1)
