"""Local shared BLE provider: exclusive control sessions and passive observers."""
import json
import os
from pathlib import Path
from queue import Empty, Full, Queue
import secrets
import socket
import subprocess
import sys
import tempfile
from threading import Lock, Thread
import time
import uuid

PORT = int(os.environ.get('VSSS_PROVIDER_PORT', '38475'))
TOKEN_PATH = Path(tempfile.gettempdir()) / ('vsss-provider-%d.token' % PORT)
IDLE_SECONDS = 5.


def stamp(kind, **values):
    return dict(type=kind, monotonic_ns=time.monotonic_ns(), wall_time_ns=time.time_ns(), **values)


class Provider:
    def __init__(self):
        self.lock = Lock()
        self.controllers = {}
        self.observers = set()
        self.clients = 0
        self.last_empty = time.monotonic()
        self.token = secrets.token_urlsafe(32)

    def publish(self, event):
        with self.lock:
            observers = tuple(self.observers)
        for observer in observers:
            try:
                observer.put_nowait(event)
            except Full:
                # End the observer rather than silently losing records.
                try:
                    observer.get_nowait()
                    observer.put_nowait(stamp('recorder_overflow', error='observer queue full'))
                    observer.put_nowait(None)
                except (Empty, Full):
                    pass
                with self.lock:
                    self.observers.discard(observer)

    def handle(self, conn):
        with self.lock:
            self.clients += 1
        conn.settimeout(5)
        reader = conn.makefile('r', encoding='utf-8', newline='\n')
        try:
            hello = json.loads(reader.readline())
            if hello.get('token') != self.token:
                return
            conn.sendall(b'{"provider":"accepted"}\n')
            conn.settimeout(None)
            role = hello.get('role')
            if role == 'observe':
                self.observe(conn)
            elif role == 'control':
                name = hello.get('name')
                if not isinstance(name, str) or not name or len(name) > 128:
                    return
                self.control(conn, reader, name)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        finally:
            reader.close()
            conn.close()
            with self.lock:
                self.clients -= 1
                if self.clients == 0:
                    self.last_empty = time.monotonic()

    def observe(self, conn):
        import select
        queue = Queue(maxsize=4096)
        with self.lock:
            self.observers.add(queue)
            cars = list(self.controllers)
        try:
            conn.sendall(b'{"provider":"observing"}\n')
            conn.sendall((json.dumps(stamp('provider_snapshot', controlled_cars=cars)) + '\n').encode())
            while True:
                try:
                    event = queue.get(timeout=.1)
                    if event is None:
                        break
                    conn.sendall((json.dumps(event) + '\n').encode())
                    if event.get('type') == 'recorder_overflow':
                        break
                except Empty:
                    pass
                ready, _, _ = select.select([conn], [], [], 0)
                if ready and not conn.recv(1, socket.MSG_PEEK):
                    break
        finally:
            with self.lock:
                self.observers.discard(queue)

    def control(self, conn, reader, name):
        session_id = uuid.uuid4().hex
        with self.lock:
            if name in self.controllers:
                conn.sendall((json.dumps({'error': 'Car is already controlled by another client', 'ready': False}) + '\n').encode())
                return
            self.controllers[name] = None
        worker = None
        try:
            worker = subprocess.Popen([sys.executable, '-u', '-m', 'vsss._ble_worker', name],
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      text=True, encoding='utf-8', bufsize=1,
                                      creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            with self.lock:
                self.controllers[name] = worker
            self.publish(stamp('control_acquired', car=name, session_id=session_id))

            def output():
                try:
                    for line in worker.stdout:
                        msg = json.loads(line)
                        if 'event' in msg:
                            self.publish(dict(msg['event'], car=name, session_id=session_id))
                        else:
                            if 'ready' in msg:
                                self.publish(stamp('connection', car=name, session_id=session_id, ready=msg['ready']))
                            if 'error' in msg:
                                self.publish(stamp('error', car=name, session_id=session_id, error=msg['error']))
                            conn.sendall(line.encode('utf-8'))
                except (OSError, ValueError):
                    pass
                finally:
                    try:
                        conn.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass

            output_thread = Thread(target=output, daemon=True)
            output_thread.start()
            for line in reader:
                worker.stdin.write(line)
                worker.stdin.flush()
        finally:
            if worker is not None:
                try:
                    worker.stdin.close()
                    worker.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    worker.kill()
                    worker.wait()
                worker.stdout.close()
            with self.lock:
                self.controllers.pop(name, None)
            self.publish(stamp('control_released', car=name, session_id=session_id))

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            listener.bind(('127.0.0.1', PORT))
            listener.listen()
            listener.settimeout(.25)
            TOKEN_PATH.write_text(self.token, encoding='ascii')
            try:
                while True:
                    try:
                        conn, _ = listener.accept()
                        Thread(target=self.handle, args=(conn,), daemon=True).start()
                    except socket.timeout:
                        with self.lock:
                            if self.clients == 0 and time.monotonic() - self.last_empty >= IDLE_SECONDS:
                                break
            finally:
                try:
                    if TOKEN_PATH.read_text(encoding='ascii') == self.token:
                        TOKEN_PATH.unlink()
                except OSError:
                    pass


def main():
    try:
        Provider().run()
    except OSError:
        pass  # Another provider already owns the endpoint.


if __name__ == '__main__':
    main()
