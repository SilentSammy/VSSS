import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
import urllib.request

import cv2
import numpy as np

from detection.cam_config import CameraInput, LatestCamera, camera_for_setup
from detection.camera_stream import publish_camera, stop_owned
from detection.frame_stream import (CameraStreamError, FramePublisher, SharedFrameReader,
                          descriptor, service_running, request_stop)


ROOT = Path(__file__).resolve().parents[1] / "detection"


class FrameTransportTests(unittest.TestCase):
    def test_stop_accepts_venv_launcher_parent_but_not_an_unrelated_pid(self):
        self.assertFalse(request_stop(self.path, -1))
        self.assertTrue(request_stop(self.path, self.publisher.info['parent_pid']))
        self.assertTrue(self.publisher.stop_requested())

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.publisher = FramePublisher(self.path)
        self.image = np.full((96, 128, 3), (7, 35, 90), np.uint8)
        self.publisher.initialize(self.image, {"kind": "synthetic"})
        self.publisher.publish(self.image, time.perf_counter_ns())
        self.readers = []

    def tearDown(self):
        for reader in self.readers:
            reader.release()
        self.publisher.close()
        self.temp.cleanup()

    def reader(self):
        reader = SharedFrameReader(self.path)
        self.readers.append(reader)
        return reader

    def test_private_copies_and_independent_reader_lifetime(self):
        first = self.reader()
        second = self.reader()
        image = first.read()
        image[:] = 0
        np.testing.assert_array_equal(second.read(), self.image)
        first.release()
        self.assertTrue(service_running(self.path))
        third = self.reader()
        np.testing.assert_array_equal(third.read(), self.image)

    def test_slow_reader_gets_latest_and_duplicate_owner_is_rejected(self):
        reader = self.reader()
        reader.read()
        for _ in range(5):
            self.publisher.publish(self.image, time.perf_counter_ns())
        reader.read()
        self.assertEqual(reader.sequence, 6)
        with self.assertRaises(CameraStreamError):
            FramePublisher(self.path)

    def test_stale_frames_are_rejected(self):
        reader = self.reader()
        self.publisher.publish(self.image, time.perf_counter_ns() - 3_000_000_000)
        with self.assertRaises(CameraStreamError):
            reader.read()

    def test_stream_failure_does_not_attempt_to_open_hardware(self):
        with patch("detection.cam_config.SharedFrameReader", side_effect=CameraStreamError("stalled")), \
                patch("detection.cam_config.LatestCamera") as hardware:
            with self.assertRaises(CameraStreamError):
                CameraInput(mode="auto", directory=self.path)
            hardware.assert_not_called()

    def test_calibration_and_rotation_are_preserved(self):
        from detection.cam_config import Camera
        template = Camera(np.eye(3), np.zeros(5), lambda: None,
                          rotation=cv2.ROTATE_90_CLOCKWISE, image_shape_hw=(96, 128),
                          device_candidates=(1, 0))
        camera, _ = camera_for_setup(template, "stream", self.path)
        try:
            np.testing.assert_array_equal(camera.K, template.K)
            self.assertEqual(camera.read()[1].shape, (128, 96, 3))
            self.assertFalse(template._released)
        finally:
            camera.release()

    def test_absent_stream_falls_back_to_direct_camera(self):
        with tempfile.TemporaryDirectory() as path, patch("detection.cam_config.LatestCamera") as hardware:
            source = CameraInput(mode="auto", directory=path)
            self.assertEqual(source.mode, "direct")
            hardware.assert_called_once()
            source.release()


class CameraServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.log = (self.path / "service.log").open("w+", encoding="utf-8")
        self.process = subprocess.Popen(
            [sys.executable, "-B", str(ROOT / "camera_stream.py"), "--synthetic",
             "--stream-dir", str(self.path), "--bind", "127.0.0.1", "--port", "0"],
            cwd=ROOT, stdout=self.log, stderr=self.log, start_new_session=sys.platform != "win32")
        deadline = time.perf_counter() + 15
        while time.perf_counter() < deadline:
            if self.process.poll() is not None:
                self.fail("Service exited during startup")
            try:
                self.info = descriptor(self.path)
                if self.info.get("ready"):
                    self.url = f"http://127.0.0.1:{self.info['http']['port']}"
                    return
            except CameraStreamError:
                pass
            time.sleep(0.05)
        self.fail("Service did not start")

    def tearDown(self):
        stop_owned(self.process, self.path)
        self.log.close()
        self.temp.cleanup()

    def test_http_original_snapshot_and_on_demand_encoding(self):
        with urllib.request.urlopen(self.url + "/health", timeout=4) as response:
            initial = json.load(response)
        self.assertTrue(initial["healthy"])
        self.assertEqual(initial["encoded_frames"], 0)
        with urllib.request.urlopen(self.url + "/snapshot.jpg", timeout=4) as response:
            image = cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape, (480, 640, 3))
        corners, ids, _ = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)).detectMarkers(image)
        self.assertIn(7, ids.flatten())
        with urllib.request.urlopen(self.url + "/camera.mjpg", timeout=4) as response:
            self.assertIn("multipart", response.headers["Content-Type"])
            prefix = response.read(128)
            self.assertIn(b"Content-Type: image/jpeg", prefix)
        self.assertIsNone(self.process.poll())

    def test_real_main_uses_stream_sends_telemetry_and_leaves_reused_service_alive(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
            udp.bind(("127.0.0.1", 0))
            port = udp.getsockname()[1]
            result = subprocess.run(
                [sys.executable, "-B", str(ROOT / "camera_stream.py"), "--run-vision",
                 "--stream-dir", str(self.path), "--headless", "--max-frames", "20",
                 "--broadcast", "127.0.0.1", "--port", str(port)],
                cwd=ROOT, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Camera source: stream", result.stdout)
            udp.settimeout(2)
            packet = json.loads(udp.recvfrom(65535)[0])
            self.assertIn("frame_sequence", packet)
            self.assertGreaterEqual(packet["frame_age_ms"], 0)
            self.assertIn("markers", packet)
        self.assertIsNone(self.process.poll())
        self.assertTrue(service_running(self.path))


class LauncherTests(unittest.TestCase):
    def test_marker_features_keep_global_coordinates_and_track_motion(self):
        from detection.marker_tracker import MarkerTracker
        image = np.full((720, 1280, 3), 255, np.uint8)
        marker = cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 4, 100)
        image[250:350, 700:800] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        corners = np.float32([[[700, 250], [799, 250], [799, 349], [700, 349]]])
        tracker = MarkerTracker()
        tracker.update(image, [corners], np.array([[4]]))
        points = tracker.tracks[4]["points"]
        self.assertGreaterEqual(points[:, :, 0].min(), 700)
        self.assertGreaterEqual(points[:, :, 1].min(), 250)
        moved = cv2.warpAffine(image, np.float32([[1, 0, 3], [0, 1, 2]]), (1280, 720),
                               borderValue=(255, 255, 255))
        tracked = tracker.update(moved, [], None)
        self.assertEqual(len(tracked), 1)
        self.assertTrue(tracked[0][2])
        np.testing.assert_allclose(tracked[0][1], corners.reshape(4, 2) + [3, 2], atol=0.6)

    def test_fullhd_calibration_accounts_for_wider_sensor_view(self):
        from detection.cam_config import Camera
        K = np.array([[735., 0, 308.], [0, 736., 242.], [0, 0, 1.]])
        frame = np.zeros((1080, 1920, 3), np.uint8)
        camera = Camera(K, np.zeros(5), lambda: frame,
                        calibration_size_wh=(640, 480), calibration_fit="expand")
        camera.read()
        # Compare projected 3D points, including distortion, between modes.
        points = np.array([[0., 0., 2.], [.2, -.1, 1.], [-.3, .2, 3.]])
        old, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), K, camera.D)
        new, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), camera.K, camera.D)
        expected = old * 2.25
        expected[:, :, 0] += 240
        np.testing.assert_allclose(new, expected)
        np.testing.assert_array_equal(K, camera._calibration_K)

    def test_capture_requests_mjpeg_before_fullhd_dimensions(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (False, None)
        with patch("detection.cam_config.cv2.VideoCapture", return_value=capture):
            source = LatestCamera((0,), 1920, 1080, 30, "MJPG")
            source.release()
        props = [call.args for call in capture.set.call_args_list]
        self.assertEqual(props[:4], [
            (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")),
            (cv2.CAP_PROP_FRAME_WIDTH, 1920), (cv2.CAP_PROP_FRAME_HEIGHT, 1080),
            (cv2.CAP_PROP_FPS, 30)])

    def test_linux_capture_falls_back_after_usb_device_numbers_change(self):
        stable = Path("/dev/v4l/by-id/usb-test-camera-video-index0")
        failed_one, failed_zero, working = Mock(), Mock(), Mock()
        failed_one.isOpened.return_value = failed_zero.isOpened.return_value = False
        working.isOpened.return_value = True
        working.read.return_value = (False, None)
        with patch("detection.cam_config.sys.platform", "linux"), \
                patch("detection.cam_config.Path.glob", side_effect=[[stable], [Path("/dev/video2")]]), \
                patch("detection.cam_config.cv2.VideoCapture", side_effect=[failed_one, failed_zero, working]) as capture:
            source = LatestCamera((1, 0))
            try:
                self.assertEqual(source.device, str(stable))
                self.assertEqual([call.args[0] for call in capture.call_args_list], [1, 0, str(stable)])
                failed_one.release.assert_called_once()
                failed_zero.release.assert_called_once()
            finally:
                source.release()

    def test_streamer_opens_selected_setup_directly_and_releases_it(self):
        """A publisher must not attach to its own stream or ignore DEFAULT_SETUP."""
        from detection import vision_setup
        default, alternate = Mock(), Mock()
        for args, selected in (([], default), (["--setup", "alternate"], alternate)):
            with self.subTest(args=args), \
                    patch.object(vision_setup, "DEFAULT_SETUP", default), \
                    patch.dict(vision_setup.VISION_SETUPS, {"alternate": alternate}, clear=True), \
                    patch("detection.camera_stream.FramePublisher") as publisher, \
                    patch("detection.camera_stream.signal.signal"):
                publisher.return_value.stop_requested.return_value = True
                self.assertEqual(publish_camera(["--no-http", *args]), 0)
                selected.camera.open_capture.assert_called_once_with(None, None, None)
                selected.camera.open_capture.return_value.release.assert_called_once()
                publisher.return_value.close.assert_called_once()

    @unittest.skipIf(sys.platform == "win32", "POSIX launcher signal handling")
    def test_sigterm_cleans_up_owned_service(self):
        self.assert_interrupt_cleans_up_owned_service(signal.SIGTERM)

    @unittest.skipIf(sys.platform == "win32", "POSIX terminal process groups")
    def test_terminal_ctrl_c_cleans_up_owned_service(self):
        self.assert_interrupt_cleans_up_owned_service(signal.SIGINT)

    def assert_interrupt_cleans_up_owned_service(self, interrupt):
        with tempfile.TemporaryDirectory() as path:
            process = subprocess.Popen(
                [sys.executable, "-B", str(ROOT / "camera_stream.py"), "--run-vision", "--synthetic-camera",
                 "--stream-dir", path, "--stream-bind", "127.0.0.1", "--stream-port", "0",
                 "--headless"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
            try:
                deadline = time.perf_counter() + 15
                while time.perf_counter() < deadline:
                    try:
                        if descriptor(path).get("ready"):
                            break
                    except CameraStreamError:
                        pass
                    self.assertIsNone(process.poll())
                    time.sleep(.05)
                else:
                    self.fail("Synthetic launcher did not start")
                # Ctrl+C targets the whole foreground group; the camera service
                # has its own session and must be stopped by the supervisor.
                os.killpg(process.pid, interrupt)
                process.wait(12)
                self.assertEqual(process.returncode, 130)
                self.assertFalse(service_running(path))
                self.assertFalse((Path(path) / "stream.json").exists())
            finally:
                stop_owned(process)

    def test_direct_capture_finishes_read_before_releasing_handle(self):
        class Capture:
            in_read = False
            released_during_read = False
            def isOpened(self): return True
            def set(self, *_): return True
            def read(self):
                self.in_read = True
                time.sleep(0.02)
                self.in_read = False
                return True, np.zeros((96, 128, 3), np.uint8)
            def release(self):
                self.released_during_read = self.in_read
        capture = Capture()
        with patch("detection.cam_config.cv2.VideoCapture", return_value=capture):
            source = LatestCamera()
            self.assertIsNotNone(source.read())
            source.release()
        self.assertFalse(capture.released_during_read)
        self.assertFalse(source.thread.is_alive())

    def test_busy_http_port_fails_without_leaving_camera_owned(self):
        with tempfile.TemporaryDirectory() as path, socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            result = subprocess.run(
                [sys.executable, "-B", str(ROOT / "camera_stream.py"), "--run-vision", "--synthetic-camera",
                 "--stream-dir", path, "--stream-bind", "127.0.0.1", "--stream-port",
                 str(occupied.getsockname()[1]), "--headless", "--max-frames", "5"],
                cwd=ROOT, capture_output=True, text=True, timeout=25)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(service_running(path))
            self.assertFalse((Path(path) / "stream.json").exists())

    def test_owned_service_is_cleaned_up_after_vision_exits(self):
        with tempfile.TemporaryDirectory() as path:
            result = subprocess.run(
                [sys.executable, "-B", str(ROOT / "camera_stream.py"), "--run-vision", "--synthetic-camera",
                 "--stream-dir", path, "--stream-bind", "127.0.0.1", "--stream-port", "0",
                 "--headless", "--max-frames", "10"], cwd=ROOT, capture_output=True,
                text=True, timeout=25)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse(service_running(path))
            self.assertFalse((Path(path) / "stream.json").exists())


if __name__ == "__main__":
    unittest.main()
