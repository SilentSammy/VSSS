import cv2
import numpy as np


class MarkerTracker:
    def __init__(self, max_missed=5, min_points=6, fb_threshold=1.0):
        self.max_missed = max_missed
        self.min_points = min_points
        self.fb_threshold = fb_threshold
        self.previous_gray = None
        self.tracks = {}

    def _reset(self, marker_id, gray, corners):
        # Work only near this marker, not over a full HD image for every ID.
        # Keep points in full-frame coordinates for subsequent optical flow.
        x, y, w, h = cv2.boundingRect(corners.astype(np.float32))
        x0, y0 = max(0, x - 3), max(0, y - 3)
        x1, y1 = min(gray.shape[1], x + w + 3), min(gray.shape[0], y + h + 3)
        points = None
        if x1 > x0 and y1 > y0:
            region = gray[y0:y1, x0:x1]
            offset = np.array([x0, y0], dtype=np.float32)
            mask = np.zeros(region.shape, dtype=np.uint8)
            cv2.fillConvexPoly(mask, (corners - offset).astype(np.int32), 255)
            points = cv2.goodFeaturesToTrack(region, 100, 0.01, 5, mask=mask)
            if points is not None:
                points += offset
        self.tracks[marker_id] = {
            "anchor_points": points,
            "points": None if points is None else points.copy(),
            "anchor_corners": corners.copy(),
            "missed": 0,
        }

    def _track(self, track, gray):
        if self.previous_gray is None or track["points"] is None:
            return None

        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray, gray, track["points"], None,
            winSize=(21, 21), maxLevel=3,
        )
        if new_points is None or status is None:
            return None

        back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.previous_gray, new_points, None,
            winSize=(21, 21), maxLevel=3,
        )
        if back_points is None or back_status is None:
            return None

        error = np.linalg.norm(
            track["points"].reshape(-1, 2) - back_points.reshape(-1, 2),
            axis=1,
        )
        good = (
            (status.ravel() == 1)
            & (back_status.ravel() == 1)
            & (error < self.fb_threshold)
        )
        anchor_points = track["anchor_points"][good]
        new_points = new_points[good]
        if len(new_points) < self.min_points:
            return None

        transform, inliers = cv2.findHomography(
            anchor_points, new_points, cv2.RANSAC, 3.0
        )
        if transform is None or inliers is None or inliers.sum() < 4:
            return None

        keep = inliers.ravel() == 1
        track["anchor_points"] = anchor_points[keep]
        track["points"] = new_points[keep]
        corners = cv2.perspectiveTransform(
            track["anchor_corners"].reshape(-1, 1, 2), transform
        ).reshape(4, 2)

        if (
            not np.all(np.isfinite(corners))
            or not cv2.isContourConvex(corners)
            or cv2.contourArea(corners) < 25
        ):
            return None

        track["missed"] += 1
        return corners

    def update(self, frame, corners, ids):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = {}

        if ids is not None:
            for corner, marker_id in zip(corners, ids.flatten()):
                marker_id = int(marker_id)
                marker_corners = corner.reshape(4, 2).astype(np.float32)
                detected[marker_id] = marker_corners
                self._reset(marker_id, gray, marker_corners)

        results = [
            (marker_id, marker_corners, False)
            for marker_id, marker_corners in detected.items()
        ]

        for marker_id, track in list(self.tracks.items()):
            if marker_id in detected:
                continue
            if track["missed"] >= self.max_missed:
                del self.tracks[marker_id]
                continue

            marker_corners = self._track(track, gray)
            if marker_corners is None:
                del self.tracks[marker_id]
            else:
                results.append((marker_id, marker_corners, True))

        self.previous_gray = gray
        return results
