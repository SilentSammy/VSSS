"""Publish camera frames locally and serve a browser view on the LAN.

Run directly to stream DEFAULT_SETUP; --setup selects another VisionSetup.
--run-vision also launches main.py and cleans up only its own processes.
Raw frames use frame_stream.py; JPEG encoding runs in a separate worker process.
"""

# Support both `python -m detection.NAME` and direct script launches.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "detection"

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from .frame_stream import (CameraStreamError, FramePublisher, SharedFrameReader,
                          default_stream_dir, descriptor, request_stop, service_running)


class JpegFrames:
    def __init__(self, reader, fps, width, quality):
        self.reader, self.fps, self.width, self.quality = reader, fps, width, quality
        self.condition = threading.Condition()
        self.viewers = 0
        self.demand_until = 0
        self.sequence = 0
        self.jpeg = None
        self.captured_ns = 0
        self.error = None
        self.stopped = False

    def encode(self):
        try:
            while not self.stopped:
                with self.condition:
                    active = self.condition.wait_for(
                        lambda: self.stopped or self.viewers or time.perf_counter() < self.demand_until,
                        timeout=1)
                if self.stopped:
                    break
                if not active:
                    continue
                started = time.perf_counter()
                frame = self.reader.read()
                if frame.shape[1] > self.width:
                    target_height = max(1, round(frame.shape[0] * self.width / frame.shape[1]))
                    frame = cv2.resize(frame, (self.width, target_height), interpolation=cv2.INTER_AREA)
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                if not ok:
                    raise CameraStreamError("JPEG encode failed")
                with self.condition:
                    self.jpeg = encoded.tobytes()
                    self.captured_ns = self.reader.last_capture_ns
                    self.sequence += 1
                    self.condition.notify_all()
                time.sleep(max(0, 1 / self.fps - (time.perf_counter() - started)))
        except BaseException as error:
            with self.condition:
                self.error = str(error)
                self.condition.notify_all()

    def next(self, previous=0):
        with self.condition:
            self.demand_until = time.perf_counter() + 2
            self.condition.notify_all()
            if not self.condition.wait_for(lambda: self.sequence != previous or self.error, timeout=3):
                raise CameraStreamError("Timed out waiting for a camera image")
            if self.error:
                raise CameraStreamError(self.error)
            return self.sequence, self.jpeg, self.captured_ns


def make_handler(frames, max_viewers=8):
    slots = threading.BoundedSemaphore(max_viewers)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.connection.settimeout(3)
            path = self.path.split("?", 1)[0]
            if path == "/health":
                info = frames.reader.status()
                info.update({"viewers": frames.viewers, "encoder_error": frames.error,
                             "encoded_frames": frames.sequence,
                             "viewer_fps": frames.fps, "viewer_max_width": frames.width})
                payload = json.dumps(info).encode()
                self.send_response(200 if info["healthy"] and not frames.error else 503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if path == "/":
                payload = (b'<!doctype html><meta name="viewport" content="width=device-width">'
                           b'<title>Camera</title><style>body{margin:0;background:#111}img{max-width:100%;height:auto}</style>'
                           b'<img src="/camera.mjpg" alt="Live camera">')
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if path not in ("/camera", "/camera.mjpg", "/stream.mjpg", "/snapshot.jpg"):
                self.send_error(404)
                return
            if not slots.acquire(blocking=False):
                self.send_error(429, "Too many camera viewers")
                return
            streaming = path != "/snapshot.jpg"
            try:
                if streaming:
                    with frames.condition:
                        frames.viewers += 1
                        frames.condition.notify_all()
                # A snapshot must be freshly encoded, not a cached idle image.
                previous = frames.sequence
                sequence, jpeg, captured = frames.next(previous)
                self.send_response(200)
                self.send_header("Cache-Control", "no-store")
                if not streaming:
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.send_header("X-Frame-Age-Ms", f"{(time.perf_counter_ns() - captured) / 1e6:.3f}")
                    self.end_headers()
                    self.wfile.write(jpeg)
                    return
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                while True:
                    headers = (f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n"
                               f"X-Frame-Sequence: {sequence}\r\n\r\n").encode()
                    self.wfile.write(headers + jpeg + b"\r\n")
                    self.wfile.flush()
                    sequence, jpeg, captured = frames.next(sequence)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError, CameraStreamError):
                pass  # A slow/disconnected viewer never stops capture or encoding.
            finally:
                if streaming:
                    with frames.condition:
                        frames.viewers -= 1
                slots.release()

    return Handler


def serve_http(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-dir", type=Path, default=default_stream_dir())
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument("--ready-file", type=Path)
    args = parser.parse_args(argv)
    if not 0 < args.fps <= 60 or args.max_width < 32 or not 1 <= args.quality <= 100:
        parser.error("Invalid stream FPS, width or quality")
    if hasattr(os, "nice"):
        os.nice(10)
    cv2.setNumThreads(1)
    reader = SharedFrameReader(args.stream_dir, timeout=10)
    frames = JpegFrames(reader, args.fps, args.max_width, args.quality)
    server = None
    try:
        server = ThreadingHTTPServer((args.bind, args.port), make_handler(frames))
        server.daemon_threads = True
        worker = threading.Thread(target=frames.encode, daemon=True)
        worker.start()
        if args.ready_file:
            args.ready_file.write_text(json.dumps({"bind": args.bind, "port": server.server_port}), encoding="utf-8")
        print(f"Camera viewers: http://{args.bind}:{server.server_port}/ (max {args.fps:g} FPS, width {args.max_width})", flush=True)
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        frames.stopped = True
        with frames.condition:
            frames.condition.notify_all()
        if server:
            server.server_close()
        if "worker" in locals():
            worker.join(4)
        reader.release()



def synthetic_frame(width, height, sequence):
    frame = np.full((height, width, 3), 230, np.uint8)
    side = max(24, min(width, height) // 4)
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 7, side)
    x = (sequence * 3) % (width - side)
    y = (height - side) // 2
    frame[y:y + side, x:x + side] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    frame.reshape(-1)[:8] = np.frombuffer(sequence.to_bytes(8, "little"), np.uint8)
    return frame


def publish_camera(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-dir", type=Path, default=default_stream_dir())
    # Import setups only at startup; cam_config never imports this module.
    from .vision_setup import DEFAULT_SETUP, VISION_SETUPS
    parser.add_argument("--setup", choices=VISION_SETUPS,
                        help="camera setup (default: DEFAULT_SETUP in vision_setup.py)")
    parser.add_argument("--run-vision", action="store_true",
                        help="also launch main.py (use --run-vision --help for launcher options)")
    parser.add_argument("--synthetic", action="store_true", help="generate test frames; never open a camera")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--camera-fps", type=float)
    parser.add_argument("--duration", type=float, help="optional test duration in seconds")
    parser.add_argument("--no-http", action="store_true")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--viewer-fps", type=float, default=10)
    parser.add_argument("--viewer-width", type=int, default=1280)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    args = parser.parse_args(argv)
    setup = VISION_SETUPS[args.setup] if args.setup else DEFAULT_SETUP
    if ((args.width is None) != (args.height is None)
            or args.width is not None and (args.width < 96 or args.height < 96)
            or args.camera_fps is not None and not 0 < args.camera_fps <= 120
            or args.duration is not None and args.duration <= 0):
        parser.error("Supply valid paired dimensions, FPS, and duration")
    if (not 0 <= args.port <= 65535 or not 0 < args.viewer_fps <= 60
            or args.viewer_width < 32 or not 1 <= args.jpeg_quality <= 100):
        parser.error("Invalid viewer port, FPS, width or JPEG quality")
    cv2.setNumThreads(1)
    stopping = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopping.set())
    publisher = None
    camera = http_process = ready_file = None
    started = time.perf_counter()
    deadline = started
    frame_number = 0
    ready = False
    announced = False
    try:
        publisher = FramePublisher(args.stream_dir)
        if not args.synthetic:
            camera = setup.camera.open_capture(args.width, args.height, args.camera_fps)
        while not stopping.is_set() and not publisher.stop_requested():
            if args.duration and time.perf_counter() - started >= args.duration:
                break
            if args.synthetic:
                time.sleep(max(0, deadline - time.perf_counter()))
                frame_number += 1
                image = synthetic_frame(args.width or 640, args.height or 480, frame_number)
                captured = time.perf_counter_ns()
                deadline = max(deadline + 1 / (args.camera_fps or 30), time.perf_counter())
            else:
                image = camera.read()
                captured = camera.last_capture_ns
                if image is None:
                    raise CameraStreamError("Camera stopped or did not deliver a frame")
            if publisher.info is None:
                source = {"kind": "synthetic"} if args.synthetic else {"kind": "camera", "device": camera.device}
                publisher.initialize(image, source)
                publisher.info.update(ready=args.no_http, http=None)
            publisher.publish(image, captured)
            if publisher.sequence == 1 and not args.no_http:
                ready_file = publisher.directory / f"{publisher.session}.http.json"
                command = [sys.executable, "-B", str(Path(__file__).resolve()), "--http-worker",
                           "--stream-dir", str(publisher.directory), "--bind", args.bind,
                           "--port", str(args.port), "--fps", str(args.viewer_fps),
                           "--max-width", str(args.viewer_width), "--quality", str(args.jpeg_quality),
                           "--ready-file", str(ready_file)]
                http_process = subprocess.Popen(command)
            if http_process is not None:
                if http_process.poll() is not None:
                    raise CameraStreamError("Camera web server exited; check its port and startup error above")
                if not ready and ready_file.is_file():
                    try:
                        http = json.loads(ready_file.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        continue  # Atomicity of this small readiness file is not assumed.
                    publisher.update_descriptor(http=http, ready=True)
                    ready = True
                if not ready and time.perf_counter() - started > 20:
                    raise CameraStreamError("Camera web server did not become ready")
            elif args.no_http:
                ready = True
            if ready and not announced:
                print(f"Camera service ready: {publisher.info['source']}, {image.shape[1]}x{image.shape[0]}; "
                      f"local frames: {publisher.directory}", flush=True)
                announced = True
    except CameraStreamError as error:
        print(f"Camera service: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        if http_process is not None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(http_process.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                http_process.terminate()
            try:
                http_process.wait(5)
            except subprocess.TimeoutExpired:
                http_process.kill()
                http_process.wait()
        if camera is not None:
            camera.release()
        if publisher is not None:
            publisher.close()
        if ready_file:
            ready_file.unlink(missing_ok=True)
    return 0



def stop_owned(process, directory=None):
    if process is None or process.poll() is not None:
        return
    if directory is not None:
        request_stop(directory, process.pid)
        try:
            process.wait(5)
            return
        except subprocess.TimeoutExpired:
            pass
    if directory is not None and os.name != "nt":
        os.killpg(process.pid, signal.SIGTERM)
    elif os.name != "nt":
        # main.py handles KeyboardInterrupt and releases its camera/windows.
        process.send_signal(signal.SIGINT)
    else:
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        process.wait(5)
    except subprocess.TimeoutExpired:
        if directory is not None and os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif directory is not None:
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def run_vision(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--camera-source", choices=("auto", "direct", "stream"), default="auto")
    parser.add_argument("--stream-dir", type=Path, default=default_stream_dir())
    parser.add_argument("--stream-bind", default="0.0.0.0")
    parser.add_argument("--stream-port", type=int, default=8080)
    parser.add_argument("--stream-fps", type=float, default=10)
    parser.add_argument("--stream-width", type=int, default=1280)
    parser.add_argument("--stream-quality", type=int, default=80)
    parser.add_argument("--no-camera-viewer", action="store_true",
                        help="disable the camera HTTP/JPEG viewer for maximum throughput")
    parser.add_argument("--synthetic-camera", action="store_true", help="test without camera hardware")
    args, main_args = parser.parse_known_args(argv)
    from . import main as vision_main
    if "--help" in main_args or "-h" in main_args:
        parser.print_help()
        vision_main.argument_parser().print_help()
        return 0
    # Validate vision options before opening hardware or launching services.
    options = vision_main.argument_parser().parse_args(main_args)
    if options.max_frames is not None and options.max_frames < 1:
        parser.error("--max-frames must be positive")
    if not 0 <= args.stream_port <= 65535 or not 0 < args.stream_fps <= 60 or args.stream_width < 32:
        parser.error("Invalid camera stream port, FPS or width")
    if not 1 <= args.stream_quality <= 100:
        parser.error("Invalid JPEG quality")
    setup = vision_main.VISION_SETUPS[options.setup] if options.setup else vision_main.DEFAULT_SETUP
    service = vision = None
    def interrupted(*_):
        raise KeyboardInterrupt()
    shutdown_signals = {signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", signal.SIGTERM)}
    previous_handlers = {sig: signal.getsignal(sig) for sig in shutdown_signals}
    for sig in shutdown_signals:
        signal.signal(sig, interrupted)
    try:
        mode = args.camera_source
        if mode != "direct" and getattr(setup.camera, "device_candidates", None) is not None:
            if not service_running(args.stream_dir):
                command = [sys.executable, "-B", str(Path(__file__).resolve()),
                           "--stream-dir", str(args.stream_dir),
                           *(["--setup", options.setup] if options.setup else []),
                           "--bind", args.stream_bind, "--port", str(args.stream_port),
                           "--viewer-fps", str(args.stream_fps), "--viewer-width", str(args.stream_width),
                           "--jpeg-quality", str(args.stream_quality)]
                if args.no_camera_viewer:
                    command.append("--no-http")
                if args.synthetic_camera:
                    command.append("--synthetic")
                service = subprocess.Popen(command, start_new_session=os.name != "nt")
            elif args.synthetic_camera:
                raise CameraStreamError("Use a separate --stream-dir for a synthetic camera")
            deadline = time.perf_counter() + 25
            while True:
                if service is not None and service.poll() is not None:
                    raise CameraStreamError("Camera service startup failed; see the error above")
                try:
                    info = descriptor(args.stream_dir)
                    if not info.get("ready"):
                        raise CameraStreamError("Camera service is starting")
                    probe = SharedFrameReader(args.stream_dir, timeout=0.2)
                    probe.release()
                    break
                except CameraStreamError:
                    if time.perf_counter() >= deadline:
                        raise CameraStreamError("Camera service did not become ready (possibly stalled)")
                    time.sleep(0.05)
            if info.get("http"):
                print(f"Camera viewer: http://localhost:{info['http']['port']}/", flush=True)
            mode = "stream"
        elif args.synthetic_camera:
            parser.error("--synthetic-camera requires a local camera service")
        command = [sys.executable, "-B", str(Path(__file__).with_name("main.py")), *main_args,
                   "--camera-source", mode, "--stream-dir", str(args.stream_dir)]
        vision = subprocess.Popen(command)
        return vision.wait()
    except KeyboardInterrupt:
        return 130
    except CameraStreamError as error:
        print(f"Camera stream: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        # A second Ctrl+C must not interrupt cleanup and leave capture running.
        for sig in shutdown_signals:
            signal.signal(sig, signal.SIG_IGN)
        try:
            try:
                stop_owned(vision)
            finally:
                stop_owned(service, args.stream_dir)
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)



def main(argv=None):
    """One entry point; worker mode is used only by the publisher itself."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--http-worker" in argv:
        argv.remove("--http-worker")
        return serve_http(argv)
    if "--run-vision" in argv:
        argv.remove("--run-vision")
        return run_vision(argv)
    return publish_camera(argv)


if __name__ == "__main__":
    raise SystemExit(main())
