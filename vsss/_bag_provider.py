"""Recorded events served through the normal local provider protocol."""
from copy import deepcopy
from threading import Event, Lock, Thread
import time
import uuid

from ._bag import BagReader
from ._provider import Provider, stamp


class BagSource:
    def __init__(self, path, publish, speed=1., loop=False, on_start=None):
        self.reader = BagReader(path, speed=speed)
        self.info = dict(self.reader.snapshot.get('vision') or {})
        self.info.setdefault('local_port', 0)
        self.info['source_id'] = uuid.uuid4().hex
        self.info['replay'] = dict(path=str(path), speed=speed, loop=loop)
        self.users = 0
        self.publish = publish
        self.loop = loop
        self.started = Event()
        self.on_start = on_start
        self.closed = Event()
        self.ended = Event()
        self.end_event = None
        self.metadata_ready = Event()
        self.metadata_ready.set()
        self.activation_lock = Lock()
        self.cycle_id = uuid.uuid4().hex
        self.thread = Thread(target=self.run, name='bag-playback', daemon=True)
        self.thread.start()

    def snapshot(self):
        return dict(self.info, replay=dict(self.info['replay'], started=self.started.is_set(),
                                          finished=self.ended.is_set()))

    def start(self, reason):
        with self.activation_lock:
            if not self.started.is_set():
                if self.on_start:
                    self.on_start(reason)
                self.started.set()

    def activate(self, queue):
        with self.activation_lock:
            if self.ended.is_set():
                queue.put_nowait(self.end_event)
        self.start('first feed subscriber connected')

    def run(self):
        try:
            self.play()
        except (OSError, ValueError, KeyError, TypeError) as error:
            with self.activation_lock:
                self.ended.set()
                self.end_event = stamp('stream_end', complete=False, error=str(error))
                self.publish(self.end_event)

    def play(self):
        while not self.closed.is_set() and not self.started.wait(.1):
            pass
        while not self.closed.is_set():
            if self.reader.finished:
                if self.loop and not self.reader.empty:
                    self.reader.rewind()
                    self.cycle_id = uuid.uuid4().hex
                    continue
                with self.activation_lock:
                    self.ended.set()
                    self.end_event = stamp('stream_end', complete=True, source_complete=self.reader.complete)
                    self.publish(self.end_event)
                break
            event = self.reader.read(.05)
            if event is None:
                continue
            event = deepcopy(event)
            event['replay'] = dict(recorded_monotonic_ns=event.get('monotonic_ns'),
                                   recorded_wall_time_ns=event.get('wall_time_ns'),
                                   recorded_source_id=event.get('source_id'))
            event.update(monotonic_ns=time.monotonic_ns(), wall_time_ns=time.time_ns())
            # Keep the fanout identity stable; each loop gets a separate timing
            # identity for GPS without altering original packet bytes.
            event['replay']['cycle_source_id'] = self.cycle_id
            self.publish(event)

    def close(self):
        self.closed.set()
        self.thread.join(timeout=1)
        self.reader.close()


class BagProvider(Provider):
    def __init__(self, path, speed=1., loop=False, immediate=False):
        super().__init__()
        self.key = 'bag'
        self.immediate = immediate
        self.source = BagSource(path, self._publish_bag, speed, loop,
                               on_start=lambda reason: print(f'Playback started: {reason}.', flush=True))
        self.selected = (self.key, self.source.info['source_id'])

    def on_listening(self):
        if self.immediate:
            self.source.start('--immediate')

    def _publish_bag(self, event):
        self.publish(event, self.selected)

    def acquire_source(self, hello):
        return self.key, self.source

    def available_cars(self, refresh=False):
        return []

    def release_source(self, key, source):
        source.users -= 1

    def control(self, conn, reader, name):
        import json
        conn.sendall((json.dumps(dict(ready=False,
            error='Bag playback is read-only; no car commands can be sent.')) + '\n').encode())

    def should_exit(self):
        # Let observation handlers drain their queues through stream_end before
        # shutting down their sockets. Immediate playback can finish without users.
        return self.source.ended.is_set() and not self.observers

    def run(self):
        try:
            super().run()
        finally:
            self.source.close()
