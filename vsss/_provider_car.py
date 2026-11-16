"""One BLE worker per car, shared by independent local client sessions."""
import json
from queue import Full, Queue
import subprocess
import sys
from threading import RLock, Thread
import uuid


class CarSession:
    def __init__(self, name, emit):
        self.emit = emit
        self.lock = RLock()
        self.clients = {}
        self.cache = {'ready': False}
        self.pending = {}
        self.velocities = {}
        self.next_request = 0
        self.next_sequence = 0
        self.owner = None
        self.closed = False
        self.worker = subprocess.Popen([sys.executable, '-u', '-m', 'vsss._ble_worker', name],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding='utf-8', bufsize=1,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.thread = Thread(target=self.output, daemon=True)
        self.thread.start()

    def attach(self):
        with self.lock:
            if self.closed:
                raise ConnectionError('Car BLE session has ended; reconnect')
            session_id = uuid.uuid4().hex
            queue = Queue(maxsize=4096)
            self.clients[session_id] = queue
            queue.put(dict(self.cache))
            self.emit('control_acquired', session_id=session_id)
            return session_id, queue

    def deliver(self, session_id, message):
        queue = self.clients.get(session_id)
        if queue is not None:
            try:
                queue.put_nowait(message)
            except Full:
                # Disconnect a stalled client instead of blocking other cars.
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait(None)

    def write(self, message):
        self.worker.stdin.write(json.dumps(message) + '\n')
        self.worker.stdin.flush()

    def set_owner(self, session_id, reason):
        if self.owner != session_id:
            previous = self.owner
            self.owner = session_id
            self.emit('motion_owner', session_id=session_id, previous_session_id=previous, reason=reason)

    def command(self, session_id, message):
        with self.lock:
            if self.closed:
                raise ConnectionError('Car BLE session has ended')
            message = dict(message, _session_id=session_id)
            if 'velocity' in message:
                self.set_owner(session_id, 'velocity')
                self.next_sequence += 1
                self.velocities[self.next_sequence] = (session_id, message['sequence'])
                message = dict(message, sequence=self.next_sequence)
            elif 'waypoint' in message or 'operation' in message:
                kind = 'waypoint' if 'waypoint' in message else 'operation'
                if kind == 'waypoint':
                    self.set_owner(session_id, 'waypoint')
                    for client in self.clients:
                        self.deliver(client, dict(motion_changed=True))
                self.next_request += 1
                self.pending[(kind, self.next_request)] = (session_id, message['request'])
                message = dict(message, request=self.next_request)
            else:
                raise ValueError('Unknown car command')
            self.write(message)

    def output(self):
        try:
            for line in self.worker.stdout:
                message = json.loads(line)
                with self.lock:
                    if 'event' in message:
                        self.emit('ble_event', event=message['event'],
                                  session_id=message['event'].get('origin_session_id'))
                        continue
                    if 'written' in message:
                        for sequence in tuple(self.velocities):
                            if sequence <= message['written']:
                                client, original = self.velocities.pop(sequence)
                                self.deliver(client, dict(written=original))
                        continue
                    for key in ('ready', 'trim_matrix', 'controller_config', 'robot_status'):
                        if key in message:
                            self.cache[key] = message[key]
                    for kind in ('operation', 'waypoint'):
                        key = kind + '_done'
                        if key in message:
                            target = self.pending.pop((kind, message[key]), None)
                            if target:
                                client, original = target
                                state = {key: message[key] for key in
                                         ('trim_matrix', 'controller_config', 'robot_status') if key in message}
                                if state:
                                    for other in self.clients:
                                        if other != client:
                                            self.deliver(other, state)
                                self.deliver(client, dict(message, **{key: original}))
                            break
                    else:
                        if 'ready' in message:
                            self.emit('connection', ready=message['ready'])
                            if not message['ready']:
                                self.set_owner(None, 'disconnected')
                        if 'error' in message:
                            self.emit('error', error=message['error'])
                        for client in self.clients:
                            self.deliver(client, message)
        except (OSError, ValueError):
            pass
        finally:
            with self.lock:
                self.closed = True
                for client in self.clients:
                    self.deliver(client, None)

    def detach(self, session_id):
        with self.lock:
            self.clients.pop(session_id, None)
            self.pending = {key: value for key, value in self.pending.items() if value[0] != session_id}
            self.velocities = {key: value for key, value in self.velocities.items() if value[0] != session_id}
            if self.owner == session_id:
                self.set_owner(None, 'client_left')
                self.next_sequence += 1
                try:
                    self.write(dict(velocity=[0, 0, 0], sequence=self.next_sequence,
                                    force=True, _session_id=session_id))
                except (OSError, ValueError):
                    pass
            self.emit('control_released', session_id=session_id)

    def close(self):
        with self.lock:
            self.closed = True
            try:
                self.write(dict(shutdown=True))
                self.worker.stdin.close()
            except OSError:
                pass
        try:
            self.worker.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.worker.kill()
            self.worker.wait()
        self.thread.join(timeout=1)
        self.worker.stdout.close()
