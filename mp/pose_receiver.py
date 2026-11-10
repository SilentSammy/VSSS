"""Small polling VSSS telemetry receiver for MicroPython ESP32 firmware."""
try:
    import ujson as json
except ImportError:
    import json
import socket
import time


class PoseReceiver:
    """Subscribe to the vision server and retain one marker's latest pose."""

    def __init__(self, marker_id, server="255.255.255.255", port=45888,
                 stale_ms=500, subscribe_ms=1000, subscribe=True):
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
        self.packets = 0
        self.parse_errors = 0
        self._next_subscribe = 0
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._socket.bind(("0.0.0.0", 0 if self.subscribe else self.port))
        self._socket.setblocking(False)

    def close(self):
        if self.subscribe:
            try:
                self._socket.sendto(b"VSSS_UNSUBSCRIBE", (self.server, self.port + 1))
            except OSError:
                pass
        self._socket.close()

    def poll(self):
        """Perform bounded network work; return True when a valid packet arrived."""
        now = time.ticks_ms()
        if self.subscribe and time.ticks_diff(now, self._next_subscribe) >= 0:
            try:
                self._socket.sendto(b"VSSS_SUBSCRIBE", (self.server, self.port + 1))
            except OSError:
                pass
            self._next_subscribe = time.ticks_add(now, self.subscribe_ms)

        updated = False
        while True:
            try:
                payload, _ = self._socket.recvfrom(4096)
            except OSError:
                break
            try:
                packet = json.loads(payload)
                self.packets += 1
                self.frame_sequence = packet.get("frame_sequence")
                self.frame_age_ms = packet.get("frame_age_ms")
                marker = None
                if packet.get("board_detected"):
                    for candidate in packet.get("markers", ()):
                        if int(candidate.get("id", -1)) == self.marker_id:
                            marker = candidate
                            break
                self.pose = (float(marker["x"]), float(marker["y"]),
                             float(marker["angle"])) if marker else None
                self.received_ms = time.ticks_ms()
                updated = True
            except (ValueError, TypeError, KeyError):
                self.parse_errors += 1
        return updated

    @property
    def fresh(self):
        return (self.pose is not None and self.received_ms is not None and
                time.ticks_diff(time.ticks_ms(), self.received_ms) <= self.stale_ms)

