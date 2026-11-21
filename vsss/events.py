"""Bounded event streams shared by all backends; callbacks stay in clients."""
from copy import deepcopy
from queue import Empty, Full, Queue
from threading import Event as Flag, RLock
import time
import uuid

from .models import Event


class EventHub:
    def __init__(self):
        self.lock = RLock()
        self.sequence = 0
        self.subscriptions = {}
        self.subscribed = Flag()
        self.on_subscribe = None

    def subscribe(self, session, topics, capacity=256):
        if not isinstance(topics, list) or not topics or any(t not in {"gps", "commands", "lifecycle"} for t in topics):
            raise ValueError("topics must be gps, commands, or lifecycle")
        if type(capacity) is not int or not 1 <= capacity <= 4096:
            raise ValueError("capacity must be 1..4096")
        identifier = uuid.uuid4().hex
        with self.lock:
            self.subscriptions[identifier] = [session, set(topics), Queue(capacity), False]
            self.subscribed.set()
        # The queue exists before playback is allowed to produce its first event.
        if self.on_subscribe:
            self.on_subscribe()
        return identifier

    def publish(self, topic, value):
        with self.lock:
            self.sequence += 1
            event = Event(topic, deepcopy(value), self.sequence, time.monotonic())
            for entry in self.subscriptions.values():
                _, topics, queue, overflow = entry
                if topic not in topics or overflow:
                    continue
                try:
                    queue.put_nowait(event)
                except Full:
                    entry[3] = True
        return event

    def poll(self, session, identifier, timeout=.2):
        if not isinstance(timeout, (int, float)) or not 0 <= timeout <= 1:
            raise ValueError("poll timeout must be 0..1 seconds")
        with self.lock:
            entry = self.subscriptions.get(identifier)
            if entry is None or entry[0] != session:
                raise ValueError("unknown subscription")
            if entry[3]:
                raise OverflowError("event subscription overflow; stream is incomplete")
        try:
            return entry[2].get(timeout=timeout)
        except Empty:
            return None

    def unsubscribe(self, session, identifier):
        with self.lock:
            if identifier in self.subscriptions and self.subscriptions[identifier][0] == session:
                del self.subscriptions[identifier]

    def disconnect(self, session):
        with self.lock:
            self.subscriptions = {key: entry for key, entry in self.subscriptions.items() if entry[0] != session}
