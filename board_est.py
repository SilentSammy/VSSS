import threading
import numpy as np
import cv2
import math
from matrix_help import ( reverse_xyz_to_zyx_4x4, extract_euler_zyx, Rx, Ry, Rz, vecs_to_matrix, matrix_to_vecs, Rx180 )

class PnpResult:
    def __init__(self, obj_pts, img_pts, tvec, rvec):
        """
        obj_pts: array of shape (N, 1, 3) or (N, 3) containing 3D object‐space coordinates
                 (X, Y, Z) of detected Charuco corners (Z is usually 0).
        img_pts: array of shape (N, 1, 2) or (N, 2) containing 2D image‐space coordinates (u, v).
        tvec, rvec: the usual solvePnP outputs (not used in project_point).
        """
        # Convert obj_pts to shape (N, 2) by flattening and taking X, Y only
        obj = np.asarray(obj_pts, dtype=np.float32)
        if obj.ndim == 3 and obj.shape[1] == 1 and obj.shape[2] == 3:
            obj = obj.reshape(-1, 3)
        elif obj.ndim == 2 and obj.shape[1] == 3:
            pass
        else:
            raise ValueError(f"Unexpected obj_pts shape {obj.shape}, expected (N,1,3) or (N,3)")

        # Only keep X, Y columns
        self.obj_pts = obj[:, :2].copy()  # shape (N, 2)

        # Convert img_pts to shape (N, 2)
        img = np.asarray(img_pts, dtype=np.float32)
        if img.ndim == 3 and img.shape[1] == 1 and img.shape[2] == 2:
            img = img.reshape(-1, 2)
        elif img.ndim == 2 and img.shape[1] == 2:
            pass
        else:
            raise ValueError(f"Unexpected img_pts shape {img.shape}, expected (N,1,2) or (N,2)")

        self.img_pts = img.copy()  # shape (N, 2)

        self.tvec = tvec
        self.rvec = rvec

    def get_quad_corners(self):
        """
        Selects four corners from obj_pts/img_pts that correspond to the board's
        outer quadrilateral. Returns (quad_obj_pts, quad_img_pts), each shape (4, 2).
        """
        N = self.obj_pts.shape[0]
        if N < 4:
            raise ValueError("Need at least 4 points to form a quadrilateral")

        xs = self.obj_pts[:, 0]
        ys = self.obj_pts[:, 1]
        min_x, max_x = float(xs.min()), float(xs.max())
        min_y, max_y = float(ys.min()), float(ys.max())

        # Define the four ideal corner positions in object space:
        targets = [
            (min_x, min_y),  # top-left
            (max_x, min_y),  # top-right
            (max_x, max_y),  # bottom-right
            (min_x, max_y),  # bottom-left
        ]

        quad_obj = []
        quad_img = []
        used_indices = set()

        for tx, ty in targets:
            diffs = self.obj_pts - np.array([tx, ty], dtype=np.float32)
            d2 = np.sum(diffs**2, axis=1)  # squared distance to each obj_pt
            idx = int(np.argmin(d2))

            if idx in used_indices:
                # If already used, pick the next closest unused
                sorted_idxs = np.argsort(d2)
                for candidate in sorted_idxs:
                    if candidate not in used_indices:
                        idx = int(candidate)
                        break

            used_indices.add(idx)
            quad_obj.append(self.obj_pts[idx])
            quad_img.append(self.img_pts[idx])

        quad_obj = np.array(quad_obj, dtype=np.float32)  # shape (4,2)
        quad_img = np.array(quad_img, dtype=np.float32)  # shape (4,2)
        return quad_obj, quad_img

    def project_point(self, point):
        """
        Projects a 2D image point (u, v) into object‐space (X, Y) by:
          1) selecting four corners via get_quad_corners()
          2) building H = getPerspectiveTransform(quad_img→quad_obj)
          3) applying H to (u, v)

        Returns:
          (X, Y) as floats.
        """
        quad_obj, quad_img = self.get_quad_corners()
        H = cv2.getPerspectiveTransform(quad_img, quad_obj)
        pts = np.array([[[point[0], point[1]]]], dtype=np.float32)  # shape (1,1,2)
        projected = cv2.perspectiveTransform(pts, H)  # shape (1,1,2)
        X = float(projected[0, 0, 0])
        Y = float(projected[0, 0, 1])
        return (X, Y)

class BoardEstimator:
    def __init__(self, board_config, K, D=None):
        self.config = board_config
        self.board = board_config.board
        self.detector = board_config.detector
        self.K = K
        self.D = D if D is not None else np.zeros(5)

    def get_board_transform(self, frame, drawing_frame=None):
        # Detect markers/corners using config's detect method
        corners, ids = self.config.detect_corners(frame, drawing_frame=drawing_frame)
        if ids is None:
            return None
        
        # Get board pose with centering offset applied before PnP
        res = get_board_pose(self.board, self.K, self.D, corners, ids, offset=self.config.center)
        if res is None:
            return None
        
        # Convert rvec, tvec to a 4x4 transformation matrix
        board_T = vecs_to_matrix(res.rvec, res.tvec)

        # Display on the drawing frame
        if drawing_frame is not None:
            rvec, tvec = matrix_to_vecs(board_T)
            rvec_string = ', '.join([str(round(math.degrees(x), 3)) for x in rvec])
            tvec_string = ', '.join([str(round(float(x), 3)) for x in tvec])
            cv2.putText(drawing_frame, f"R: {rvec_string}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(drawing_frame, f"T: {tvec_string}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return board_T, res

def get_board_pose(
    board: cv2.aruco.Board,
    K: np.ndarray,
    D: np.ndarray,
    detected_corners: np.ndarray,
    detected_ids: np.ndarray,
    offset: np.ndarray = None
) -> PnpResult:
    """
    Estimate board pose for either CharucoBoard or GridBoard.
    
    Args:
        offset: Optional 3D offset to apply to object points before solving PnP.
                Use this to recenter the coordinate system (e.g., board center).
    """
    obj_pts, img_pts = board.matchImagePoints(detected_corners, detected_ids)
    if obj_pts.shape[0] < 6:
        return None

    # Apply offset to object points before solving PnP
    if offset is not None:
        obj_pts = obj_pts - offset

    success, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        K,
        D,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    return PnpResult(obj_pts=obj_pts, img_pts=img_pts, tvec=tvec.flatten(), rvec=rvec.flatten())

# Legacy functions for direct use with CharucoBoard and GridBoard
def get_charuco_pose(
    board: cv2.aruco.CharucoBoard,
    K: np.ndarray,
    D: np.ndarray,
    charuco_corners: np.ndarray,
    charuco_ids: np.ndarray,
    center: bool = False
) -> PnpResult:
    """
    Estimate the Charuco‐board pose, optionally recentering the translation
    so that the board’s center is treated as the origin instead of its top-left corner.
    """
    obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
    if obj_pts.shape[0] < 6:
        return None

    if center:
        # Compute the board's geometric center in board coords
        squaresX, squaresY = board.getChessboardSize()
        sq_len = board.getSquareLength()
        center_board = np.array([
            ((squaresX - 1) * sq_len / 2.0) + sq_len / 2.0,
            (squaresY - 1) * sq_len / 2.0 + sq_len / 2.0,
            0.0
        ], dtype=np.float64)
        # Subtract the center from all obj_pts
        obj_pts = obj_pts - center_board

    success, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        K,
        D,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    return PnpResult(obj_pts=obj_pts, img_pts=img_pts, tvec=tvec.flatten(), rvec=rvec.flatten())

def get_grid_pose(
    board: cv2.aruco.GridBoard,
    K: np.ndarray,
    D: np.ndarray,
    marker_corners: np.ndarray,
    marker_ids: np.ndarray,
    center: bool = False
) -> PnpResult:
    """
    Estimate the GridBoard pose, optionally recentering the translation
    so that the board's center is treated as the origin instead of its top-left corner.
    """
    obj_pts, img_pts = board.matchImagePoints(marker_corners, marker_ids)
    if obj_pts.shape[0] < 6:
        return None

    if center:
        # Compute the board's geometric center in board coords
        gridX, gridY = board.getGridSize()
        marker_len = board.getMarkerLength()
        marker_sep = board.getMarkerSeparation()
        # Total size = (n_markers * marker_len) + ((n_markers - 1) * separation)
        total_x = (gridX * marker_len) + ((gridX - 1) * marker_sep)
        total_y = (gridY * marker_len) + ((gridY - 1) * marker_sep)
        center_board = np.array([
            total_x / 2.0,
            total_y / 2.0,
            0.0
        ], dtype=np.float64)
        # Subtract the center from all obj_pts
        obj_pts = obj_pts - center_board

    success, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        K,
        D,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    return PnpResult(obj_pts=obj_pts, img_pts=img_pts, tvec=tvec.flatten(), rvec=rvec.flatten())
