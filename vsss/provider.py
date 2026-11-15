"""Passive observation of the active live, bag or simulation provider."""
import json
import select
import socket
import time

from ._provider_connection import connect


class ProviderClient:
    """Observe selected topics. Disconnection/overflow explicitly ends the stream.

    Each instance has one reader. Call close() from another thread to interrupt it.
    Reconnection is a consumer policy: recorders fail incomplete; GPS reconnects.
    """
    def __init__(self, topics=('ble', 'udp'), vision=None):
        self.socket = connect('observe', topics=list(topics), vision=vision or {})
        self.buffer = b''
        self.finished = False
        try:
            self.socket.setblocking(False)
            hello = self.read(5)
            if hello != {'provider': 'observing'}:
                raise ConnectionError('Provider rejected observation settings: ' + str(hello))
            self.snapshot = self.read(5)
            if not self.snapshot or self.snapshot.get('type') != 'provider_snapshot':
                raise ConnectionError('Provider did not supply its snapshot')
        except Exception:
            self.close()
            raise

    def read(self, timeout=.2):
        if self.finished:
            return None
        deadline = time.monotonic() + timeout
        while b'\n' not in self.buffer:
            remaining = max(0., deadline - time.monotonic())
            ready, _, _ = select.select([self.socket], [], [], remaining)
            if not ready:
                return None
            chunk = self.socket.recv(65536)
            if not chunk:
                raise ConnectionError('VSSS provider disconnected; observation stream incomplete')
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b'\n', 1)
        event = json.loads(line)
        if event.get('type') == 'stream_end':
            self.finished = True
        return event

    def close(self):
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
