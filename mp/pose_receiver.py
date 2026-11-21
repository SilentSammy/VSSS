"""Small polling pose-telemetry receiver for MicroPython firmware."""
try:
    import ujson as json
except ImportError:
    import json
import socket
import time
import math
import gc


class PoseReceiver:
    """Subscribe to the vision server and retain one marker's latest pose."""

    def __init__(self, marker_id, server="192.168.12.1", port=5000,
                 stale_ms=800, subscribe_ms=1000, subscribe=True):
        self.marker_id = int(marker_id)
        self.server = server
        self.port = int(port)
        self.stale_ms = int(stale_ms)
        self.subscribe_ms = int(subscribe_ms)
        self.subscribe = bool(subscribe)
        self.pose = None
        self.received_ms = None
        self.frame_sequence = None
        self.frame_age_ms = None
        self.stream_id = None
        self.sequence_gap = None
        self.receive_gap_ms = None
        self.packets = 0
        self.parse_errors = 0
        self.memory_errors = 0
        self._receive_failed = False
        self._next_subscribe = 0
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._socket.bind(("0.0.0.0", 0 if self.subscribe else self.port))
        self._socket.setblocking(False)

    def close(self):
        if self.subscribe:
            try:
                self._socket.sendto(b"POSE_UNSUBSCRIBE", (self.server, self.port + 1))
            except OSError:
                pass
        self._socket.close()

    def poll(self):
        """Perform bounded network work; return True when a valid packet arrived."""
        now = time.ticks_ms()
        if self.subscribe and time.ticks_diff(now, self._next_subscribe) >= 0:
            try:
                self._socket.sendto(b"POSE_SUBSCRIBE", (self.server, self.port + 1))
            except OSError:
                pass
            self._next_subscribe = time.ticks_add(now, self.subscribe_ms)

        updated = False
        for _ in range(8):
            try:
                payload, _ = self._socket.recvfrom(4096)
            except MemoryError:
                self.memory_errors += 1
                self._receive_failed = True
                gc.collect()
                break
            except OSError:
                break
            try:
                packet = json.loads(payload)
                if not isinstance(packet, dict):
                    raise ValueError('packet must be an object')
                self.packets += 1
                sequence = packet.get('frame_sequence')
                stream_id = packet.get('stream_id')
                frame_age = packet.get('frame_age_ms')
                if sequence is not None and (type(sequence) is not int or sequence < 0):
                    raise ValueError('invalid frame sequence')
                if frame_age is not None:
                    frame_age = float(frame_age)
                    if not math.isfinite(frame_age) or frame_age < 0:
                        raise ValueError('invalid frame age')
                if (sequence is not None and self.frame_sequence is not None and
                        stream_id == self.stream_id and sequence <= self.frame_sequence):
                    continue  # Replays/duplicates never renew position freshness.
                marker = None
                if packet.get("board_detected"):
                    for candidate in packet.get("markers", ()):
                        if int(candidate.get("id", -1)) == self.marker_id:
                            marker = candidate
                            break
                pose = (float(marker["x"]), float(marker["y"]),
                        float(marker["angle"])) if marker else None
                if pose is not None and not all(math.isfinite(v) for v in pose):
                    raise ValueError('invalid pose')
                accepted_ms = time.ticks_ms()
                if stream_id == self.stream_id and self.received_ms is not None:
                    self.receive_gap_ms = max(0, time.ticks_diff(accepted_ms, self.received_ms))
                    self.sequence_gap = (sequence - self.frame_sequence
                                         if sequence is not None and self.frame_sequence is not None else None)
                else:
                    self.receive_gap_ms = self.sequence_gap = None
                self.pose = pose
                self.frame_sequence, self.frame_age_ms = sequence, frame_age
                self.stream_id = stream_id
                self.received_ms = accepted_ms
                self._receive_failed = False
                updated = True
            except MemoryError:
                self.memory_errors += 1
                self._receive_failed = True
                gc.collect()
                break
            except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
                self.parse_errors += 1
        return updated

    @property
    def age_ms(self):
        if self.received_ms is None:
            return None
        return max(0, time.ticks_diff(time.ticks_ms(), self.received_ms)) + (self.frame_age_ms or 0)

    @property
    def fresh(self):
        age = self.age_ms
        return (not self._receive_failed and self.pose is not None and
                age is not None and age <= self.stale_ms)

