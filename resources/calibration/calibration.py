import cv2
import numpy as np
import glob

# — CONFIG — 
CHECKERBOARD = (8, 5)           # number of inner corners per row, column
square_size  = 0.03             # actual square size in meters (or any unit)

# termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
            30,   # max iterations
            1e-6) # epsilon

# storage for object points and image points
obj_points = []  # 3D points in real world
img_points = []  # 2D points in image plane

# prepare one pattern of object points, e.g. (0,0,0), (1,0,0), ..., (8,5,0)
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# — LOAD IMAGES & FIND CORNERS —
images = [img for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff'] 
          for img in glob.glob(rf'resources\calibration\{ext}')]

for fname in images:
    img = cv2.imread(fname)

    # rotate 90
    # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find checkerboard corners
    ok, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                            cv2.CALIB_CB_ADAPTIVE_THRESH 
                                          + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        print(f"⚠️ Corners not found in {fname}")
        continue

    # refine to subpixel accuracy
    corners_refined = cv2.cornerSubPix(gray, corners, (3,3), (-1,-1), criteria)

    img_points.append(corners_refined)
    obj_points.append(objp)

    # (optional) draw and display:
    cv2.drawChessboardCorners(img, CHECKERBOARD, corners_refined, ok)
    cv2.imshow('Corners', img)
    cv2.waitKey(100)

cv2.destroyAllWindows()

# — RUN FISHEYE CALIBRATION —
N_OK = len(obj_points)
if N_OK < 10:
    print(f"❌ Only {N_OK} valid patterns; need 10–20 good shots.")
    exit()

K = np.zeros((3,3))
D = np.zeros((4,1))
rvecs = [np.zeros((1,1,3), dtype=np.float64) for _ in range(N_OK)]
tvecs = [np.zeros((1,1,3), dtype=np.float64) for _ in range(N_OK)]

# calibration flags — fix skew, assume principal point at center, etc.
flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
       + cv2.fisheye.CALIB_CHECK_COND
       + cv2.fisheye.CALIB_FIX_SKEW)

rms, _, _, _, _ = cv2.fisheye.calibrate(
    obj_points,
    img_points,
    gray.shape[::-1],
    K, D,
    rvecs, tvecs,
    flags,
    criteria
)

print(f"Calibration done with RMS error = {rms:.6f}")
print("Intrinsic matrix K:\n", K)
print("Distortion coefficients D:\n", D.ravel())