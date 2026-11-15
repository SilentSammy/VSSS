"""Shared local BLE control and original UDP/BLE observation streams."""
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

from ._provider_vision import VisionSource, source_settings

PORT = int(os.environ.get('VSSS_PROVIDER_PORT', '38475'))
TOKEN_PATH = Path(tempfile.gettempdir()) / ('vsss-provider-%d.token' % PORT)
IDLE_SECONDS = 5.


def stamp(kind, **values):
    return dict(type=kind, monotonic_ns=time.monotonic_ns(), wall_time_ns=time.time_ns(), **values)


class Provider:
    def __init__(self):
        self.lock = Lock()
        self.publish_lock = Lock()
        self.controllers = {}
        self.observers = set()
        self.sources = {}
        self.clients = 0
        self.last_empty = time.monotonic()
        self.token = secrets.token_urlsafe(32)

    def acquire_source(self, hello):
        """Called under the provider lock; sources share one observation contract."""
        key = source_settings(hello.get('vision', {})) if 'udp' in hello.get('topics', ()) else None
        source = self.sources.get(key) if key is not None else None
        if key is not None and source is None:
            source = VisionSource(key, lambda event: self.publish(event, (key, event['source_id'])))
            self.sources[key] = source
        return key, source

    def release_source(self, key, source):
        source.users -= 1
        if source.users == 0:
            self.sources.pop(key)
            source.close()

    def should_exit(self):
        return self.clients == 0 and time.monotonic() - self.last_empty >= IDLE_SECONDS

    def publish(self, event, source=None):
        with self.publish_lock:
            self._publish(event, source)

    def _publish(self, event, source):
        with self.lock:
            observers = tuple(self.observers)
        for observer, topics, selected in observers:
            if event['type'] in ('udp', 'stream_end'):
                if event['type'] == 'stream_end':
                    if selected != source:
                        continue
                elif 'udp' not in topics or selected != source:
                    continue
            elif 'ble' not in topics:
                continue
            try:
                observer.put_nowait(event)
            except Full:
                # End the observer rather than silently losing records.
                try:
                    while True:
                        observer.get_nowait()
                except Empty:
                    pass
                observer.put_nowait(stamp('observer_overflow', error='observer queue full; stream incomplete'))
                with self.lock:
                    self.observers.discard((observer, topics, selected))

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
                self.observe(conn, hello)
            elif role == 'control':
                name = hello.get('name')
                if not isinstance(name, str) or not name or len(name) > 128:
                    return
                self.control(conn, reader, name)
        except (OSError, ValueError, TypeError) as error:
            try:
                conn.sendall((json.dumps({'provider': 'error', 'error': str(error)}) + '\n').encode())
            except OSError:
                pass
        finally:
            reader.close()
            conn.close()
            with self.lock:
                self.clients -= 1
                if self.clients == 0:
                    self.last_empty = time.monotonic()

    def observe(self, conn, hello):
        import select
        conn.settimeout(2)  # A stalled consumer must not hold a session forever.
        queue = Queue(maxsize=4096)
        topics = frozenset(hello.get('topics', ['ble']))
        if not topics or not topics <= {'udp', 'ble'}:
            raise ValueError('Unknown observation topics')
        with self.lock:
            key, source = self.acquire_source(hello)
            if source:
                source.users += 1
            selected = (key, source.info['source_id']) if source else None
            entry = (queue, topics, selected)
            self.observers.add(entry)
            cars = list(self.controllers)
        try:
            if source:
                source.metadata_ready.wait(1.2)
            conn.sendall(b'{"provider":"observing"}\n')
            conn.sendall((json.dumps(stamp('provider_snapshot', controlled_cars=cars,
                                         vision=source.snapshot() if source else None)) + '\n').encode())
            if source and hasattr(source, 'activate'):
                source.activate(queue)
            while True:
                try:
                    event = queue.get(timeout=.1)
                    if event is None:
                        break
                    conn.sendall((json.dumps(event) + '\n').encode())
                    if event.get('type') in ('observer_overflow', 'stream_end'):
                        break
                except Empty:
                    pass
                ready, _, _ = select.select([conn], [], [], 0)
                if ready and not conn.recv(1, socket.MSG_PEEK):
                    break
        finally:
            with self.lock:
                self.observers.discard(entry)
                if source:
                    self.release_source(key, source)

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
                            if self.should_exit():
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
