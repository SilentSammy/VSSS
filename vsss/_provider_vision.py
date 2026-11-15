"""UDP ownership inside the provider; datagrams are never reconstructed."""
import base64
import json
import math
import socket
from threading import Event, Thread
import time
import uuid
from urllib.error import HTTPError
from urllib.request import build_opener, ProxyHandler

METADATA_POLL_SECONDS = 5.


def source_settings(options):
    """Canonical, hashable source identity shared by all local consumers."""
    server = options.get('server', '192.168.12.1')
    port = options.get('port', 5000)
    listen = options.get('listen_port', 0 if server else port)
    subscription = options.get('subscription_port', port + 1) if server else 0
    bind = options.get('bind', '0.0.0.0')
    metadata_url = options.get('metadata_url')
    if metadata_url is not None and (not isinstance(metadata_url, str) or
                                     not metadata_url.startswith(('http://', 'https://'))):
        raise ValueError('Invalid metadata URL')
    if not isinstance(bind, str) or (server is not None and not isinstance(server, str)):
        raise ValueError('Invalid vision address')
    for value in (port, listen, subscription):
        if type(value) is not int or not 0 <= value <= 65535:
            raise ValueError('Invalid vision port')
    if server and not subscription:
        raise ValueError('Subscription port must be positive')
    return server, port, listen, subscription, bind, metadata_url


def decode_packet(data):
    """Validate the world-state fields while retaining the original JSON object."""
    packet = json.loads(data)
    if type(packet['board_detected']) is not bool:
        raise ValueError('board_detected must be a boolean')
    if not isinstance(packet['markers'], list) or not isinstance(packet.get('balls', []), list):
        raise ValueError('markers and balls must be lists')
    for marker in packet['markers']:
        int(marker['id'])
        if not all(math.isfinite(float(marker[k])) for k in ('x', 'y', 'angle')):
            raise ValueError('Nonfinite marker')
    for ball in packet.get('balls', []):
        if not all(math.isfinite(float(ball[k])) for k in ('x', 'y')):
            raise ValueError('Nonfinite ball')
    for key in ('timestamp', 'frame_age_ms'):
        if packet.get(key) is not None:
            value = float(packet[key])
            if not math.isfinite(value) or (key == 'frame_age_ms' and value < 0):
                raise ValueError('Invalid packet timing')
    return packet


class VisionSource:
    def __init__(self, settings, publish):
        self.server, self.port, listen, self.subscription_port, bind, metadata_url = settings
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self.socket.bind((bind, listen))
            self.socket.settimeout(.2)
        except Exception:
            self.socket.close()
            raise
        self.info = dict(source_id=uuid.uuid4().hex, local_port=self.socket.getsockname()[1],
                         server=self.server, port=self.port, subscription_port=self.subscription_port,
                         bind=bind)
        self.users = 0
        self.publish = publish
        self.closed = Event()
        self.metadata_ready = Event()
        self.metadata = dict(status=None, error='metadata not fetched', fetched_at=None)
        self.metadata_url = metadata_url or (f'http://{self.server}:8080/api/v1/status'
                                             if self.server and self.server != '255.255.255.255' else None)
        self.sequence = 0
        self.target = (self.server, self.subscription_port) if self.server else None
        self.thread = Thread(target=self.run, name='provider-vision', daemon=True)
        self.thread.start()
        self.metadata_thread = Thread(target=self.poll_metadata, name='provider-metadata', daemon=True)
        self.metadata_thread.start()

    def poll_metadata(self):
        opener = build_opener(ProxyHandler({}))
        while not self.closed.is_set():
            if self.metadata_url:
                try:
                    try:
                        response = opener.open(self.metadata_url, timeout=1.)
                    except HTTPError as error:
                        if error.code != 503:
                            raise
                        response = error  # Unhealthy services still supply useful config.
                    with response:
                        status = json.loads(response.read(1024 * 1024 + 1))
                    if not isinstance(status, dict) or status.get('schema') != 'aruco-gps.status/v1':
                        raise ValueError('unsupported vision status schema')
                    self.metadata = dict(status=status, error=None, fetched_at=time.time(),
                                         url=self.metadata_url)
                except (OSError, ValueError) as error:
                    self.metadata = dict(self.metadata, error=str(error), attempted_at=time.time())
            else:
                self.metadata = dict(status=None, error='metadata URL unavailable', fetched_at=None)
                if self.closed.wait(.1):
                    break
                continue
            self.metadata_ready.set()
            if self.closed.wait(METADATA_POLL_SECONDS):
                break

    def snapshot(self):
        return dict(self.info, metadata=self.metadata)

    def run(self):
        renewal = 0.
        while not self.closed.is_set():
            if self.target and time.monotonic() >= renewal:
                try:
                    self.socket.sendto(b'POSE_SUBSCRIBE', self.target)
                except OSError:
                    pass
                renewal = time.monotonic() + 1.
            try:
                data, sender = self.socket.recvfrom(65535)
                event = dict(type='udp', monotonic_ns=time.monotonic_ns(), wall_time_ns=time.time_ns(),
                             source_id=self.info['source_id'], sequence=self.sequence,
                             sender=list(sender), payload_base64=base64.b64encode(data).decode('ascii'))
                self.sequence += 1
                try:
                    event['packet'] = decode_packet(data)
                    event['valid'] = True
                    if not self.metadata_url:
                        self.metadata_url = f'http://{sender[0]}:8080/api/v1/status'
                    if self.server and sender[1] == self.subscription_port:
                        self.target = sender
                except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError) as error:
                    event.update(valid=False, error=str(error))
                self.publish(event)
            except (socket.timeout, ConnectionResetError):
                continue
            except OSError:
                break

    def close(self):
        self.closed.set()
        if self.target:
            try:
                self.socket.sendto(b'POSE_UNSUBSCRIBE', self.target)
            except OSError:
                pass
        self.socket.close()
