"""Local provider connection and race-safe detached startup."""
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from threading import Event, Thread
import time

def connect(role, name=None, timeout=8, **options):
    port = int(os.environ.get('VSSS_PROVIDER_PORT', '38475'))
    token_path = Path(tempfile.gettempdir()) / ('vsss-provider-%d.token' % port)
    deadline = time.monotonic() + timeout
    started = False
    while time.monotonic() < deadline:
        sock = None
        try:
            token = token_path.read_text(encoding='ascii')
            sock = socket.create_connection(('127.0.0.1', port), timeout=.5)
            sock.sendall((json.dumps(dict(token=token, role=role, name=name, **options)) + '\n').encode())
            reply = b''
            while not reply.endswith(b'\n'):
                part = sock.recv(1)
                if not part or len(reply) > 1024:
                    raise ConnectionError('VSSS provider handshake failed')
                reply += part
            if json.loads(reply) != {'provider': 'accepted'}:
                raise ConnectionError('VSSS provider rejected connection')
            sock.settimeout(None)
            return sock
        except (OSError, ValueError):
            if sock:
                sock.close()
            if not started:
                # DETACHED_PROCESS makes Windows ignore CREATE_NO_WINDOW. In a
                # venv the Python launcher can then give the real interpreter
                # a new console. NO_WINDOW alone also survives that launcher.
                flags = (getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0) |
                         getattr(subprocess, 'CREATE_NO_WINDOW', 0)) if os.name == 'nt' else 0
                process = subprocess.Popen([sys.executable, '-u', '-m', 'vsss._provider'],
                                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL, creationflags=flags,
                                           start_new_session=(os.name != 'nt'), close_fds=True)
                Thread(target=process.wait, daemon=True).start()
                started = True
            time.sleep(.05)
    raise ConnectionError('Could not start or connect to the local VSSS provider')


class _Writer:
    def __init__(self, sock):
        self.sock = sock
        self.file = sock.makefile('w', encoding='utf-8', newline='\n')
    def write(self, data): return self.file.write(data)
    def flush(self): return self.file.flush()
    def close(self):
        self.file.close()
        try:
            self.sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass


class ProviderConnection:
    def __init__(self, name):
        self.sock = connect('control', name)
        self.stdin = _Writer(self.sock)
        self.stdout = self.sock.makefile('r', encoding='utf-8', newline='\n')
        self.done = Event()
    def poll(self): return 0 if self.done.is_set() else None
    def mark_done(self): self.done.set()
    def wait(self, timeout=None):
        if not self.done.wait(timeout):
            raise subprocess.TimeoutExpired('BLE provider session', timeout)
        self.sock.close()
        return 0
    def kill(self):
        try: self.sock.shutdown(socket.SHUT_RDWR)
        except OSError: pass
        self.sock.close()
        self.done.set()
