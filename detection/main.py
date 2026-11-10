# Support both `python -m detection.NAME` and direct script launches.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "detection"

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from .marker_tracker import MarkerTracker
from .cam_config import camera_for_setup
from .frame_stream import CameraStreamError, default_stream_dir
from .vision_setup import DEFAULT_SETUP, VISION_SETUPS
from .ball_det import BallDetector
from .telemetry import TelemetryPublisher

DEFAULT_TELEMETRY_PORT = 45888


def localize_balls(detections, estimator, board_result, frame, height):
    """Return ball centres in the same Cartesian metre frame as markers."""
    if board_result is None:
        return []
    _, pnp = board_result
    balls = []
    for ball in detections:
        x, y = estimator.project_point_to_board(pnp, ball.centroid, frame.shape, z=height)
        if math.isfinite(x) and math.isfinite(y):
            balls.append({"x": round(float(x), 3), "y": round(float(-y), 3)})
    return balls
# Captured from the lab desktop. Values are x, y, client width, client height;
# OpenCV positions account for the desktop's frame/border offsets.
DEFAULT_WINDOW_LAYOUT = {
    "Camera": (52, 20, 1300, 708),
    "Board": (1369, 20, 548, 1011),
}


def load_marker_config(path, *, strict=False):
    """Return validated appearance and height overrides; errors use defaults."""
    appearances, heights = {}, {}
    if path is None:
        return appearances, heights
    try:
        path = Path(path).resolve()
        with path.open(encoding="utf-8-sig") as file:
            entries = json.load(file)
        if not isinstance(entries, dict):
            raise ValueError("expected an object keyed by marker ID")
    except (OSError, UnicodeError, ValueError, RuntimeError, RecursionError) as error:
        if strict:
            raise
        print(f"Marker config unavailable; using defaults: {error}")
        return appearances, heights

    def number(value, minimum, inclusive=True):
        if type(value) not in (int, float):
            return False
        try:
            return math.isfinite(value) and (value >= minimum if inclusive else value > minimum)
        except OverflowError:
            return False

    for key, entry in entries.items():
        try:
            marker_id = int(key)
            if marker_id < 0 or not isinstance(entry, dict):
                raise ValueError("expected a nonnegative ID and an object")
        except (TypeError, ValueError):
            print(f"Ignoring invalid marker config entry: {key!r}")
            continue

        def ignored(field):
            print(f"Marker {marker_id}: invalid {field}; using its default")

        if "height_m" in entry:
            if number(entry["height_m"], 0):
                heights[marker_id] = entry["height_m"]
            else:
                ignored("height_m")
        style = {}
        if "text" in entry:
            text = entry["text"]
            if isinstance(text, str) and all(c == "\n" or " " <= c <= "~" for c in text):
                style["text"] = text
            else:
                ignored("text")
        if "color" in entry:
            color = entry["color"]
            if (isinstance(color, list) and len(color) == 3
                    and all(type(c) is int and 0 <= c <= 255 for c in color)):
                style["color"] = tuple(color)
            else:
                ignored("color")
        if "image" in entry:
            try:
                if not isinstance(entry["image"], str) or not entry["image"]:
                    raise ValueError("expected an image path")
                image_path = Path(entry["image"])
                if not image_path.is_absolute():
                    image_path = path.parent / image_path
                # Load now so a broken image cannot interrupt the tracking loop.
                image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                if (image is None or image.dtype not in (np.uint8, np.uint16)
                        or not (image.ndim == 2 or image.ndim == 3 and image.shape[2] in (3, 4))):
                    raise ValueError("unsupported image")
                style["image"] = image
            except (OSError, ValueError, cv2.error) as error:
                print(f"Marker {marker_id}: cannot load image; using label defaults: {error}")
        if "scale" in entry:
            if number(entry["scale"], 0, inclusive=False):
                style["scale"] = entry["scale"]
            else:
                ignored("scale")
        if style:
            appearances[marker_id] = style
    return appearances, heights


class MarkerConfigWatcher:
    """Check metadata twice per second; replace settings only after a valid read.

    Initial failures use defaults. Later failures retain the last good settings,
    so temporary/partial editor saves cannot interrupt tracking.
    """

    def __init__(self, path):
        self.path = Path(path) if path is not None else None
        self.current = ({}, {})
        self.signature = object()
        self.next_check = 0.0

    def poll(self, now=None):
        now = time.monotonic() if now is None else now
        if self.path is None or now < self.next_check:
            return False
        self.next_check = now + 0.5
        try:
            stat = self.path.stat()
            signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
        except OSError:
            signature = None
        if signature == self.signature:
            return False
        self.signature = signature
        try:
            updated = load_marker_config(self.path, strict=True)
        except (OSError, UnicodeError, ValueError, RuntimeError, RecursionError) as error:
            print(f"Marker config not updated; keeping previous settings: {error}", flush=True)
            return False
        self.current = updated
        print(f"Marker config loaded: {self.path}", flush=True)
        return True


def apply_marker_appearances(plotter, defaults, appearances):
    """Rebuild from defaults so deleting a JSON override really removes it."""
    overrides = {marker_id: dict(style) for marker_id, style in defaults.items()}
    for marker_id, style in appearances.items():
        overrides.setdefault(marker_id, {}).update(style)
    plotter.marker_overrides = overrides
    # Discard sprites holding earlier text/colors/images and avoid cache growth.
    plotter.marker_images.clear()


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ball-height", type=float, default=0.02,
                        help="ball centre height above the board in metres (default: 0.02)")
    parser.add_argument("--camera-source", choices=("auto", "direct", "stream"), default="auto",
                        help="prefer the local service, capture directly, or require the service")
    parser.add_argument("--stream-dir", type=Path, default=default_stream_dir(),
                        help="local camera service runtime directory")
    parser.add_argument("--headless", action="store_true", help="run without camera/board windows")
    parser.add_argument("--no-ball-detection", action="store_true",
                        help="skip ball segmentation/localization for maximum marker throughput")
    parser.add_argument("--max-frames", type=int, help="stop after this many processed frames (testing)")
    for name, geometry in DEFAULT_WINDOW_LAYOUT.items():
        parser.add_argument(f"--{name.lower()}-window", type=int, nargs=4,
                            metavar=("X", "Y", "WIDTH", "HEIGHT"), default=geometry,
                            help=f"{name} window layout (default: {' '.join(map(str, geometry))})")
    parser.add_argument(
        "--marker-config", type=Path,
        help="override the selected vision setup's marker config JSON",
    )
    parser.add_argument(
        "--setup",
        choices=VISION_SETUPS,
        help="vision setup (uses DEFAULT_SETUP from vision_setup.py if omitted)",
    )
    parser.add_argument(
        "--broadcast",
        default=None,
        help="telemetry broadcast address; telemetry is not sent if omitted",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_TELEMETRY_PORT,
        help=f"telemetry UDP port (default: {DEFAULT_TELEMETRY_PORT})",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=10.0,
        help="telemetry send rate in Hz (default: 10)",
    )
    return parser


def main(argv=None):
    parser = argument_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.ball_height) or args.ball_height < 0:
        parser.error("--ball-height must be finite and nonnegative")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    for geometry in (args.camera_window, args.board_window):
        if geometry[2] < 1 or geometry[3] < 1:
            parser.error("Window width and height must be positive")
    setup = VISION_SETUPS[args.setup] if args.setup is not None else DEFAULT_SETUP

    if args.broadcast is None:
        print("No --broadcast address given; telemetry will not be sent.")
    telemetry_interval = 1 / args.rate if args.rate > 0 else float("inf")

    try:
        camera, frame_source = camera_for_setup(setup.camera, args.camera_source, args.stream_dir)
    except CameraStreamError as error:
        print(f"Camera unavailable: {error}", flush=True)
        return 1
    if frame_source:
        print(f"Camera source: {frame_source.mode}", flush=True)
    estimator = setup.board_estimator
    detector = setup.detector
    plotter = None if args.headless else setup.plotter
    config_path = args.marker_config if args.marker_config is not None else setup.marker_config_path
    config_watcher = MarkerConfigWatcher(config_path)
    config_watcher.poll()
    appearances, marker_heights = config_watcher.current
    default_appearances = ({marker_id: dict(style) for marker_id, style in plotter.marker_overrides.items()}
                           if plotter is not None else {})
    if plotter is not None:
        apply_marker_appearances(plotter, default_appearances, appearances)
    tracker = MarkerTracker()
    ball_detector = None if args.no_ball_detection else BallDetector()

    try:
        udp = TelemetryPublisher(args.broadcast, args.port)
    except OSError:
        camera.release()
        raise

    stats_at = last_plot = time.monotonic()
    next_telemetry = stats_at + telemetry_interval
    frames = read_time = vision_time = plot_time = 0.0
    processed_frames = 0

    try:
        if not args.headless:
            cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Board", cv2.WINDOW_NORMAL)
        while True:
            started = time.monotonic()
            if config_watcher.poll(started):
                appearances, marker_heights = config_watcher.current
                if plotter is not None:
                    apply_marker_appearances(plotter, default_appearances, appearances)
            success, frame = camera.read()
            if not success:
                print("Camera stopped or could not be opened.")
                break
            read_done = time.monotonic()

            # Camera updates intrinsics from the actual capture size, including
            # the sensor mode's field of view and any configured rotation.
            estimator.K = camera.K
            estimator.D = camera.D
            board_result = estimator.get_board_transform(frame)
            corners, ids, _ = detector.detectMarkers(frame)
            tracked_markers = tracker.update(frame, corners, ids)
            # Detect before drawing coloured overlays, so they cannot become balls.
            if ball_detector is None:
                ball_detections, balls = (), []
            else:
                ball_detections = ball_detector.detect(frame)
                balls = localize_balls(ball_detections, estimator, board_result, frame, args.ball_height)
            markers = []
            positions = []
            telemetry_markers = []

            for marker_id, image_corners, is_tracked in tracked_markers:
                if board_result is not None:
                    _, pnp = board_result
                    marker_height = marker_heights.get(marker_id, setup.marker_height_m)
                    board_corners = np.array([
                        estimator.project_point_to_board(
                            pnp, point, frame.shape, marker_height
                        )
                        for point in image_corners
                    ])
                    markers.append((marker_id, board_corners, False))
                    x, y = board_corners.mean(axis=0)
                    # Heading is the outward normal of the marker's top edge.
                    # The reported frame is Cartesian: +X is 0 degrees, +Y is
                    # 90 degrees, and positive rotation is counterclockwise.
                    forward = (board_corners[0] + board_corners[1]) / 2 - (x, y)
                    angle = np.degrees(np.arctan2(-forward[1], forward[0]))
                    positions.append(
                        f"ID {marker_id} ({x:.2f}, {-y:.2f}) m {angle:.0f} deg"
                    )
                    telemetry_markers.append(
                        {"id": marker_id, "x": round(x, 3), "y": round(-y, 3), "angle": round(angle, 0)}
                    )

                color = (0, 255, 255) if is_tracked else (0, 255, 0)
                cv2.polylines(frame, [image_corners.astype(np.int32)], True, color, 2)
                marker_text = str(marker_id)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(
                    marker_text, font, font_scale, thickness
                )
                center_x, center_y = image_corners.mean(axis=0).astype(int)
                text_at = (
                    center_x - text_width // 2,
                    center_y + (text_height - baseline) // 2,
                )
                cv2.putText(
                    frame, marker_text, text_at, font, font_scale,
                    color, thickness, cv2.LINE_AA,
                )

            vision_done = time.monotonic()
            for ball in ball_detections:
                cv2.drawContours(frame, [ball.contour], -1, (0, 255, 255), 2)
            if not args.headless:
                cv2.imshow("Camera", frame)
            if plotter is not None and (processed_frames == 0 or vision_done - last_plot >= 0.1):
                cv2.imshow("Board", plotter.render(markers, balls, ball_radius=args.ball_height))
                last_plot = vision_done
            if not args.headless and processed_frames == 0:
                for name, geometry in (("Camera", args.camera_window), ("Board", args.board_window)):
                    x, y, width, height = geometry
                    cv2.resizeWindow(name, width, height)
                    try:
                        cv2.moveWindow(name, x, y)
                    except cv2.error as error:
                        print(f"Cannot position {name} window on this display backend: {error}", flush=True)
            if args.broadcast is not None and vision_done >= next_telemetry:
                telemetry = {
                    "timestamp": round(time.time(), 3),
                    "board_detected": board_result is not None,
                    "markers": telemetry_markers,
                    "balls": balls,
                }
                if frame_source:
                    telemetry.update(
                        frame_sequence=frame_source.sequence,
                        frame_age_ms=round((time.perf_counter_ns() - frame_source.last_capture_ns) / 1e6, 3),
                    )
                udp.send(telemetry)
                # Keep a steady schedule: small timing jitter around 10 FPS
                # must not turn a requested 10 Hz stream into 5 Hz.
                next_telemetry = max(next_telemetry + telemetry_interval, vision_done)
            plot_done = time.monotonic()

            frames += 1
            read_time += read_done - started
            vision_time += vision_done - read_done
            plot_time += plot_done - vision_done
            if plot_done - stats_at >= 1.0:
                print(
                    f"{frames / (plot_done - stats_at):.1f} FPS | "
                    f"board {'yes' if board_result else 'no'} | "
                    f"{', '.join(positions) or 'no markers'} | balls={balls}",
                    flush=True,
                )
                stats_at = plot_done
                frames = read_time = vision_time = plot_time = 0.0

            processed_frames += 1
            if args.max_frames and processed_frames >= args.max_frames:
                break
            if not args.headless and cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    except CameraStreamError as error:
        print(f"Camera stream lost: {error}. Restart to reconnect or use --camera-source direct.", flush=True)
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        udp.close()
        camera.release()
        if not args.headless:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
