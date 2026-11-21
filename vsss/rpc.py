"""One Pyro runtime for arbitrary marked component methods and event streams."""
from contextlib import contextmanager
import hmac
import inspect
import json
import logging
import os
from pathlib import Path
import secrets
import tempfile
from threading import Event, RLock, Thread, current_thread
import uuid

import Pyro5.api as pyro

from .components import session_context
from .models import encode, decode

log = logging.getLogger(__name__)


class RemoteError(RuntimeError):
    """A backend exception not part of the common builtin error vocabulary."""


def default_endpoint():
    return Path(tempfile.gettempdir()) / f"vsss-{os.environ.get('USERNAME', os.environ.get('USER', 'user'))}-rpc.json"


class _Daemon(pyro.Daemon):
    def __init__(self, endpoint, secret, port):
        self.endpoint, self.secret = endpoint, secret
        super().__init__(host="127.0.0.1", port=port)

    def validateHandshake(self, connection, data):
        if not isinstance(data, str) or not hmac.compare_digest(data, self.secret):
            raise ValueError("provider authentication failed")
        return "accepted"

    def clientDisconnect(self, connection):
        session = getattr(connection, "vsss_session", None)
        if session:
            try:
                self.endpoint.disconnect(session)
            except Exception:
                log.exception("session disconnect cleanup failed")


class _Endpoint:
    """The only Pyro-exposed object. Domain objects never import Pyro."""
    def __init__(self, provider):
        self.provider = provider
        self.sessions = set()
        self.lock = RLock()

    @pyro.expose
    def open(self):
        connection = pyro.current_context.client
        if getattr(connection, "vsss_session", None):
            return connection.vsss_session
        with self.lock:
            session = uuid.uuid4().hex
            self.sessions.add(session)
            connection.vsss_session = session
        return session

    def _check(self, session):
        with self.lock:
            if session not in self.sessions:
                raise ConnectionError("session has ended; reconnect explicitly")

    @pyro.expose
    def invoke(self, session, component, method, args, kwargs):
        self._check(session)
        target = {"provider": self.provider, "gps": self.provider.gps, "cars": self.provider.cars}.get(component)
        if target is None or not isinstance(method, str):
            raise ValueError("unknown component or method")
        function = inspect.getattr_static(type(target), method, None)
        if not getattr(function, "rpc_exposed", False):
            raise ValueError("method is not exposed")
        token = session_context.set(session)
        try:
            try:
                return {"ok": True, "result": encode(getattr(target, method)(*decode(args), **decode(kwargs)))}
            except Exception as error:
                # Third-party exceptions (e.g. Bleak) need not be serializable.
                return {"ok": False, "error": type(error).__name__, "message": str(error)}
        finally:
            session_context.reset(token)

    @pyro.expose
    def subscribe(self, session, topics, capacity):
        self._check(session)
        return self.provider.events.subscribe(session, topics, capacity)

    @pyro.expose
    def poll(self, session, subscription):
        self._check(session)
        return encode(self.provider.events.poll(session, subscription))

    @pyro.expose
    def unsubscribe(self, session, subscription):
        self.provider.events.unsubscribe(session, subscription)

    def disconnect(self, session):
        with self.lock:
            self.sessions.discard(session)
        self.provider.events.disconnect(session)
        self.provider.cars.disconnect(session)


class RPCHost:
    def __init__(self, provider, *, endpoint=None, port=38475):
        self.provider = provider
        self.path = Path(endpoint) if endpoint else default_endpoint()
        self.secret = secrets.token_hex(32)
        self.endpoint = _Endpoint(provider)
        self.daemon = _Daemon(self.endpoint, self.secret, port)
        self.uri = str(self.daemon.register(self.endpoint))
        self.thread = None

    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
        self.thread = Thread(target=self.daemon.requestLoop, daemon=True)
        self.thread.start()
        temporary.write_text(json.dumps({"uri": self.uri, "token": self.secret}), encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(self.path)
        return self

    def close(self):
        # Domain resources stay alive until disconnect handlers have stopped cars.
        self.daemon.shutdown()
        if self.thread:
            self.thread.join(timeout=5)
        try:
            if json.loads(self.path.read_text(encoding="utf-8")).get("token") == self.secret:
                self.path.unlink()
        except (OSError, ValueError):
            pass

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.close()


def _proxy(settings):
    proxy = pyro.Proxy(settings["uri"])
    proxy._pyroHandshake = settings["token"]
    proxy._pyroTimeout = settings.get('timeout', 30)
    return proxy


class _RemoteComponent:
    def __init__(self, client, name):
        self._client, self._name = client, name
        self._callback = None

    def __getattr__(self, method):
        if method.startswith("_"):
            raise AttributeError(method)
        return lambda *args, **kwargs: self._client._invoke(self._name, method, args, kwargs)

    @property
    def on_change(self):
        return self._callback

    @on_change.setter
    def on_change(self, callback):
        self._client._set_callback(self._name, callback)
        self._callback = callback


class Subscription:
    """Poll events on a private connection; callback failures remain inspectable."""
    def __init__(self, client, topics, callback, capacity):
        self.error = None
        self._stop = Event()
        self._client = client
        self._id = client._root_call("subscribe", client._session, topics, capacity)

        def run():
            try:
                with _proxy(client._settings) as proxy:
                    while not self._stop.is_set():
                        event = decode(proxy.poll(client._session, self._id))
                        if event is not None:
                            callback(event)
            except Exception as error:
                if not self._stop.is_set():
                    self.error = error
                    log.debug("event subscription ended", exc_info=True)
            finally:
                try:
                    with _proxy(client._settings) as proxy:
                        proxy.unsubscribe(client._session, self._id)
                except Exception:
                    pass

        self._thread = Thread(target=run, daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        if current_thread() is not self._thread:
            self._thread.join(timeout=31)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class ProviderClient:
    """One generic client: component methods, records and callbacks are shared."""
    def __init__(self, endpoint=None, *, timeout=30):
        self._settings = json.loads(Path(endpoint or default_endpoint()).read_text(encoding="utf-8"))
        self._settings['timeout'] = timeout
        self._root = _proxy(self._settings)
        self._root_lock = RLock()
        self._closed = False
        try:
            self._session = self._root.open()
        except Exception:
            self._root._pyroRelease()
            raise
        self.gps = _RemoteComponent(self, "gps")
        self.cars = _RemoteComponent(self, "cars")
        self._callbacks = {}
        self._subscriptions = []

    def _invoke(self, component, method, args, kwargs):
        response = self._root_call("invoke", self._session, component, method, encode(args), encode(kwargs))
        if not response["ok"]:
            errors = {kind.__name__: kind for kind in (ValueError, TypeError, RuntimeError, ConnectionError,
                                                       TimeoutError, NotImplementedError)}
            error = errors.get(response["error"], RemoteError)
            raise error(f"{response['error']}: {response['message']}")
        return decode(response["result"])

    def _root_call(self, name, *args):
        # A callback is allowed to issue commands from its background thread.
        with self._root_lock:
            if self._closed:
                raise RuntimeError("provider client is closed")
            self._root._pyroClaimOwnership()
            return getattr(self._root, name)(*args)

    def __getattr__(self, method):
        if method.startswith("_"):
            raise AttributeError(method)
        return lambda *args, **kwargs: self._invoke("provider", method, args, kwargs)

    def subscribe(self, topics, callback, capacity=256):
        subscription = Subscription(self, topics, callback, capacity)
        self._subscriptions.append(subscription)
        return subscription

    def _set_callback(self, component, callback):
        previous = self._callbacks.pop(component, None)
        if previous:
            previous.close()
        if callback is not None:
            if not callable(callback):
                raise TypeError("callback must be callable or None")
            topic = "gps" if component == "gps" else "commands"
            self._callbacks[component] = self.subscribe([topic], lambda event: callback(event.value))

    def close(self):
        for subscription in self._subscriptions:
            subscription.close()
        with self._root_lock:
            self._closed = True
            self._root._pyroClaimOwnership()
            self._root._pyroRelease()  # EOF releases this session's motion ownership.

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
