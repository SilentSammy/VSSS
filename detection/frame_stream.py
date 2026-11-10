"""Local latest-frame transport shared by independently launched processes.

Linux uses a memory-backed runtime directory. Explicit mmap/file locks avoid
SharedMemory resource trackers unlinking a running service's buffers.
"""

from contextlib import contextmanager
import errno
import getpass
import json
import mmap
import os
from pathlib import Path
import struct
import tempfile
import threading
import time
import uuid

import numpy as np


HEADER_BYTES = 128
HEADER = struct.Struct("<8sIIIQQB")  # magic, H/W/C, sequence, capture ns, closed
MAGIC = b"CAMIPC01"


class CameraStreamError(RuntimeError):
    pass


def default_stream_dir():
    identity = str(os.getuid()) if hasattr(os, "getuid") else getpass.getuser()
    # GUI and SSH sessions must discover the same default directory.
    user_runtime = Path("/run/user") / identity
    base = user_runtime if user_runtime.is_dir() else (
        "/dev/shm" if Path("/dev/shm").is_dir() else tempfile.gettempdir())
    return Path(base) / f"vsss-camera-{identity}"


class FileMutex:
    def __init__(self, path):
        self.path = Path(path)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        self.file = os.fdopen(fd, "r+b", buffering=0)
        if os.fstat(fd).st_size == 0:
            self.file.write(b"\0")
        self.thread_lock = threading.Lock()
        self.held = False

    def acquire(self, timeout=1):
        deadline = time.perf_counter() + timeout
        if not self.thread_lock.acquire(timeout=max(0, timeout)):
            return False
        try:
            while True:
                try:
                    if os.name == "nt":
                        import msvcrt
                        self.file.seek(0)
                        msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.held = True
                    return True
                except OSError as error:
                    if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    if time.perf_counter() >= deadline:
                        self.thread_lock.release()
                        return False
                    time.sleep(0.001)
        except BaseException:
            self.thread_lock.release()
            raise

    def release(self):
        if not self.held:
            return
        if os.name == "nt":
            import msvcrt
            self.file.seek(0)
            msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.file, fcntl.LOCK_UN)
        self.held = False
        self.thread_lock.release()

    @contextmanager
    def locked(self, timeout=1):
        if not self.acquire(timeout):
            raise CameraStreamError(f"Timed out waiting for camera lock: {self.path}")
        try:
            yield
        finally:
            self.release()

    def close(self):
        self.release()
        self.file.close()


def service_running(directory):
    directory = Path(directory)
    if not (directory / "owner.lock").is_file():
        return False
    mutex = FileMutex(directory / "owner.lock")
    try:
        available = mutex.acquire(0)
        return not available
    finally:
        mutex.close()


def descriptor(directory):
    directory = Path(directory)
    try:
        info = json.loads((directory / "stream.json").read_text(encoding="utf-8"))
        if info["version"] != 1 or not isinstance(info["session"], str):
            raise ValueError("Unsupported stream descriptor")
        for key in ("frames", "lock", "stop"):
            if Path(info[key]).name != info[key] or info[key] in (".", ".."):
                raise ValueError("Invalid stream file name")
        return info
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise CameraStreamError(f"Camera stream is not ready: {error}") from error


class FramePublisher:
    def __init__(self, directory=None):
        self.directory = Path(directory or default_stream_dir())
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.owner = FileMutex(self.directory / "owner.lock")
        if not self.owner.acquire(0):
            self.owner.close()
            raise CameraStreamError("A camera source already owns this stream directory")
        self.session = uuid.uuid4().hex
        self.info = None
        self.mapping = self.file = self.mutex = self.pixels = None
        self.sequence = 0

    def initialize(self, image, source):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise CameraStreamError("Camera service requires uint8 BGR frames")
        self.info = {
            "version": 1, "session": self.session, "pid": os.getpid(), "parent_pid": os.getppid(),
            "shape": list(image.shape), "source": source,
            "frames": f"{self.session}.frames", "lock": f"{self.session}.lock",
            "stop": f"{self.session}.stop",
        }
        fd = os.open(self.directory / self.info["frames"], os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        self.file = os.fdopen(fd, "r+b", buffering=0)
        self.file.truncate(HEADER_BYTES + image.nbytes)
        self.mapping = mmap.mmap(self.file.fileno(), 0)
        self.pixels = np.ndarray(image.shape, np.uint8, buffer=self.mapping, offset=HEADER_BYTES)
        self.mutex = FileMutex(self.directory / self.info["lock"])

    def publish(self, image, captured_ns):
        if self.info is None or list(image.shape) != self.info["shape"] or image.dtype != np.uint8:
            raise CameraStreamError("Camera frame size/format changed; restart the camera service")
        with self.mutex.locked():
            np.copyto(self.pixels, image)
            self.sequence += 1
            HEADER.pack_into(self.mapping, 0, MAGIC, *image.shape, self.sequence, captured_ns, 0)
        if self.sequence == 1:
            self.update_descriptor()

    def stop_requested(self):
        return self.info and (self.directory / self.info["stop"]).exists()

    def update_descriptor(self, **fields):
        self.info.update(fields)
        temporary = self.directory / f"{self.session}.json.tmp"
        temporary.write_text(json.dumps(self.info), encoding="utf-8")
        deadline = time.perf_counter() + 1
        while True:
            try:
                temporary.replace(self.directory / "stream.json")
                break
            except PermissionError:
                if time.perf_counter() >= deadline:
                    raise
                time.sleep(0.005)  # A Windows reader can briefly prevent replacement.

    def close(self):
        if self.mapping is not None:
            with self.mutex.locked():
                fields = list(HEADER.unpack_from(self.mapping))
                fields[-1] = 1
                HEADER.pack_into(self.mapping, 0, *fields)
            self.pixels = None
            self.mapping.close()
            self.file.close()
            self.mutex.close()
        try:
            if descriptor(self.directory)["session"] == self.session:
                (self.directory / "stream.json").unlink(missing_ok=True)
        except CameraStreamError:
            pass
        if self.info:
            for key in ("frames", "lock", "stop"):
                try:
                    (self.directory / self.info[key]).unlink(missing_ok=True)
                except PermissionError:
                    pass  # Windows readers may still hold an old mapping briefly.
        self.owner.close()


class SharedFrameReader:
    def __init__(self, directory=None, timeout=3):
        self.directory = Path(directory or default_stream_dir())
        self.mapping = self.file = self.mutex = self.pixels = None
        self.sequence = 0
        self.last_capture_ns = 0
        self.delivery_ms = 0.0
        deadline = time.perf_counter() + timeout
        while True:
            try:
                if not service_running(self.directory):
                    raise CameraStreamError("No running camera service")
                self.info = descriptor(self.directory)
                self.file = (self.directory / self.info["frames"]).open("r+b", buffering=0)
                shape = tuple(self.info["shape"])
                if (len(shape) != 3 or shape[2] != 3 or any(type(x) is not int or x <= 0 for x in shape)
                        or np.prod(shape) > 128 * 1024 * 1024):
                    raise CameraStreamError("Invalid camera frame dimensions")
                if os.fstat(self.file.fileno()).st_size != HEADER_BYTES + int(np.prod(shape)):
                    raise CameraStreamError("Camera frame buffer size is invalid")
                self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
                self.pixels = np.ndarray(shape, np.uint8, buffer=self.mapping, offset=HEADER_BYTES)
                self.mutex = FileMutex(self.directory / self.info["lock"])
                # Verify a complete, fresh first frame before declaring readiness.
                self.read(timeout=min(1, max(0.01, deadline - time.perf_counter())))
                self.sequence = 0
                return
            except (OSError, CameraStreamError, KeyError, TypeError, ValueError) as error:
                self.release()
                if time.perf_counter() >= deadline:
                    raise CameraStreamError(str(error)) from error
                time.sleep(0.02)

    def read(self, timeout=1):
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            with self.mutex.locked(timeout=min(0.1, max(0, deadline - time.perf_counter()))):
                magic, height, width, channels, sequence, captured, closed = HEADER.unpack_from(self.mapping)
                if magic != MAGIC or (height, width, channels) != self.pixels.shape or closed:
                    raise CameraStreamError("Camera service stopped or its buffer is invalid")
                if sequence != self.sequence:
                    if time.perf_counter_ns() - captured > 2_000_000_000:
                        raise CameraStreamError("Camera service is publishing stale frames")
                    image = self.pixels.copy()
                    self.sequence, self.last_capture_ns = sequence, captured
                    self.delivery_ms = (time.perf_counter_ns() - captured) / 1e6
                    return image
            # Poll a tiny header; never queue frames or spin continuously.
            time.sleep(0.001)
        raise CameraStreamError("Camera service stopped delivering new frames")

    def release(self):
        self.pixels = None
        if self.mapping is not None:
            self.mapping.close()
        if self.file is not None:
            self.file.close()
        if self.mutex is not None:
            self.mutex.close()
        self.mapping = self.file = self.mutex = None

    def status(self):
        with self.mutex.locked():
            magic, h, w, c, sequence, captured, closed = HEADER.unpack_from(self.mapping)
        age_ms = (time.perf_counter_ns() - captured) / 1e6
        return {"healthy": magic == MAGIC and not closed and age_ms < 2000,
                "sequence": sequence, "frame_age_ms": round(age_ms, 3),
                "shape": [h, w, c], "source": self.info["source"]}


def request_stop(directory, expected_pid):
    try:
        info = descriptor(directory)
        # Windows venv python.exe may be a launcher whose child runs Python.
        if expected_pid in (info["pid"], info.get("parent_pid")):
            (Path(directory) / info["stop"]).touch()
            return True
    except CameraStreamError:
        pass
    return False
