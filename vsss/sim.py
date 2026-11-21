"""Local simulator editing API. Never starts a provider or controls live hardware."""
import json
from threading import Lock

from ._provider_connection import connect


class SimClient:
    """Connect to the active simulator; positions are metres, angles degrees CCW.

    Optional source_id binds this connection to a particular observation feed.
    Balls use stable zero-based indices for the lifetime of that simulator.
    Grabs hold an object against physics until release or client disconnection.
    """
    def __init__(self, *, source_id=None, timeout=1):
        self.socket = connect('simulation', source_id=source_id, auto_start=False, timeout=timeout)
        self.socket.settimeout(timeout)
        self.reader = self.socket.makefile('r', encoding='utf-8')
        self.lock = Lock()
        try:
            reply = self._read()
            if not reply.get('ready') or (source_id is not None and reply.get('source_id') != source_id):
                raise ConnectionError(reply.get('error', 'Simulation instance changed'))
            self.source_id = reply['source_id']
        except BaseException:
            self.close()
            raise

    def _read(self):
        line = self.reader.readline(65537)
        if not line or len(line) > 65536:
            raise ConnectionError('Simulation editing connection ended')
        return json.loads(line)

    def request(self, operation, **settings):
        data = (json.dumps(dict(operation=operation, **settings), allow_nan=False) + '\n').encode()
        with self.lock:
            self.socket.sendall(data)
            reply = self._read()
        if not reply.get('ok'):
            raise ValueError(reply.get('error', 'Simulation edit rejected'))
        return reply['state']

    def state(self):
        return self.request('state')

    def move_car(self, number, x, y, angle=None):
        settings = {} if angle is None else dict(angle=angle)
        return self.request('move', kind='car', id=number, x=x, y=y, **settings)

    def move_ball(self, index, x, y):
        return self.request('move', kind='ball', id=index, x=x, y=y)

    def grab(self, kind, number):
        return self.request('grab', kind=kind, id=number)

    def release(self, kind, number):
        return self.request('release', kind=kind, id=number)

    def close(self):
        import socket
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.reader.close()
        self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
