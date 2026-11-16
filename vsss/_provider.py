"""Shared local BLE control and original UDP/BLE observation streams."""
import json
import os
from pathlib import Path
from queue import Empty, Full, Queue
import secrets
import socket
import tempfile
from threading import Lock, Thread
import time

from ._provider_vision import VisionSource, source_settings

PORT = int(os.environ.get('VSSS_PROVIDER_PORT', '38475'))
TOKEN_PATH = Path(tempfile.gettempdir()) / ('vsss-provider-%d.token' % PORT)
IDLE_SECONDS = 5.


def stamp(kind, **values):
    return dict(type=kind, monotonic_ns=time.monotonic_ns(), wall_time_ns=time.time_ns(), **values)


class Provider:
    def __init__(self):
        self.lock = Lock()
        self.car_lock = Lock()
        self.publish_lock = Lock()
        self.controllers = {}
        self.observers = set()
        self.sources = {}
        self.clients = 0
        self.connections = set()
        self.handlers = set()
        self.last_empty = time.monotonic()
        self.token = secrets.token_urlsafe(32)
        self.scan_lock = Lock()
        self.scan_time = None
        self.scan_cars = []

    def available_cars(self, refresh=False):
        """Return advertised cars plus cars already connected through this provider."""
        import asyncio
        import re
        from bleak import BleakScanner

        with self.scan_lock:
            if refresh or self.scan_time is None or time.monotonic() - self.scan_time > 30:
                devices = asyncio.run(BleakScanner.discover(timeout=4, return_adv=True))
                self.scan_cars = sorted({int(match.group(1))
                    for device, advertisement in devices.values()
                    if (match := re.fullmatch(r'car(\d+)', advertisement.local_name or device.name or ''))})
                self.scan_time = time.monotonic()
            with self.lock:
                connected = {int(match.group(1)) for name in self.controllers
                             if (match := re.fullmatch(r'car(\d+)', name))}
                return sorted(set(self.scan_cars) | connected)

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

    def on_listening(self):
        """Called once the provider endpoint is ready to accept clients."""

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
            elif role == 'discover':
                try:
                    result = dict(cars=self.available_cars(hello.get('refresh', False)))
                except Exception as error:
                    result = dict(error=f'Car discovery failed: {error}')
                conn.sendall((json.dumps(result) + '\n').encode())
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
        from ._provider_car import CarSession

        def emit(kind, **values):
            if kind == 'ble_event':
                event = dict(values.pop('event'), car=name, **values)
            else:
                event = stamp(kind, car=name, **values)
            self.publish(event)

        with self.car_lock:
            session = self.controllers.get(name)
            if session is None:
                session = CarSession(name, emit)
                self.controllers[name] = session
            session_id, queue = session.attach()

        def output():
            try:
                while True:
                    message = queue.get()
                    if message is None:
                        break
                    conn.sendall((json.dumps(message) + '\n').encode())
            except OSError:
                pass
            finally:
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

        thread = Thread(target=output, daemon=True)
        thread.start()
        try:
            for line in reader:
                session.command(session_id, json.loads(line))
        finally:
            with self.car_lock:
                session.detach(session_id)
                if not session.clients:
                    self.controllers.pop(name, None)
                    session.close()
            queue.put_nowait(None)
            thread.join(timeout=1)

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            listener.bind(('127.0.0.1', PORT))
            listener.listen()
            listener.settimeout(.25)
            TOKEN_PATH.write_text(self.token, encoding='ascii')
            try:
                self.on_listening()
                while True:
                    try:
                        conn, _ = listener.accept()
                        def serve_client(client=conn):
                            try:
                                self.handle(client)
                            finally:
                                with self.lock:
                                    self.connections.discard(client)

                        thread = Thread(target=serve_client, daemon=True)
                        with self.lock:
                            self.connections.add(conn)
                            self.handlers = {handler for handler in self.handlers if handler.is_alive()}
                            self.handlers.add(thread)
                        thread.start()
                    except socket.timeout:
                        with self.lock:
                            if self.should_exit():
                                break
            finally:
                # Closing client sessions makes control handlers stop their BLE
                # workers before an explicitly started provider exits.
                with self.lock:
                    connections = tuple(self.connections)
                    handlers = tuple(self.handlers)
                for conn in connections:
                    try:
                        conn.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                for handler in handlers:
                    handler.join(timeout=6)
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
