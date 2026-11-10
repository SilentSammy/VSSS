"""Broadcast discovery, then per-student UDP unicast for reliable WiFi delivery."""
import json
import socket
import time

SUBSCRIBE = b'VSSS_SUBSCRIBE'
UNSUBSCRIBE = b'VSSS_UNSUBSCRIBE'


class TelemetryPublisher:
    def __init__(self, broadcast, port):
        self.destination = (broadcast, port) if broadcast else None
        self._subscribers = {}
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.broadcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self.socket.bind(('', port + 1))
            self.socket.setblocking(False)
        except Exception:
            self.socket.close()
            self.broadcast_socket.close()
            raise

    def send(self, packet):
        now = time.monotonic()
        # Bound work per frame, so discovery cannot starve camera processing.
        for _ in range(64):
            try:
                message, address = self.socket.recvfrom(64)
            except BlockingIOError:
                break
            except ConnectionResetError:
                continue  # A student's closed UDP port on a Windows server.
            if message == SUBSCRIBE and (address in self._subscribers or len(self._subscribers) < 256):
                self._subscribers[address] = now + 5
            elif message == UNSUBSCRIBE:
                self._subscribers.pop(address, None)
        payload = json.dumps(packet).encode()
        for address, expires in list(self._subscribers.items()):
            if expires <= now:
                del self._subscribers[address]
                continue
            try:
                self.socket.sendto(payload, address)
            except OSError:
                pass  # A departing laptop must not interrupt everyone else's telemetry.
        if self.destination:
            self.broadcast_socket.sendto(payload, self.destination)

    def close(self):
        self.socket.close()
        self.broadcast_socket.close()
