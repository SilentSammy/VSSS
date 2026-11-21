"""Mouse gestures and asynchronous transport for a simulator-backed plot."""
from collections import deque
import math
from threading import Condition, Thread

from vsss.sim import SimClient


class EditTransport:
    """Serialize edits off the GUI thread, coalescing intermediate mouse moves."""
    def __init__(self, source_id):
        self.source_id = source_id
        self.condition = Condition()
        self.pending = deque()
        self.closed = False
        self.client = None
        self.error = None
        self.failed = False
        self.thread = Thread(target=self.run, name='plot-sim-editor', daemon=True)
        self.thread.start()

    def send(self, message):
        with self.condition:
            if self.closed or self.failed:
                return
            if (message['operation'] == 'move' and self.pending and
                    self.pending[-1]['operation'] == 'move' and
                    (self.pending[-1]['kind'], self.pending[-1]['id']) == (message['kind'], message['id'])):
                self.pending[-1] = message
            elif len(self.pending) < 128:
                self.pending.append(message)
            else:
                # Ending the connection releases any held object on the server.
                self.error = 'Too many queued edit gestures; reconnect the plot to edit again'
                self.closed = True
            self.condition.notify()

    def run(self):
        try:
            with SimClient(source_id=self.source_id) as client:
                self.client = client
                while True:
                    with self.condition:
                        self.condition.wait_for(lambda: self.closed or self.pending)
                        if self.closed:
                            return
                        message = self.pending.popleft()
                    try:
                        client.request(**message)
                    except ValueError as error:
                        self.error = str(error)
        except (OSError, ConnectionError, ValueError) as error:
            self.error = str(error)
        finally:
            self.failed = True

    def close(self):
        with self.condition:
            self.closed = True
            self.pending.clear()
            self.condition.notify()
        self.thread.join(timeout=1.2)
        if self.thread.is_alive() and self.client is not None:
            # Interrupt an outstanding reply before joining again.
            import socket
            try:
                self.client.socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.thread.join(timeout=1.2)


class SimDrag:
    def __init__(self, plotter, body, radius, send):
        self.plotter, self.body, self.radius, self.send = plotter, body, radius, send
        self.packet = None
        self.active = None
        self.center_radius = min(abs(a[0] * b[1] - a[1] * b[0]) /
                                 math.hypot(b[0] - a[0], b[1] - a[1])
                                 for a, b in zip(body, body[1:] + body[:1])) * .55

    def world_point(self, x, y):
        return ((x - self.plotter.width / 2) / self.plotter.scale,
                -(y - self.plotter.height / 2) / self.plotter.scale)

    def pixels(self, x, y):
        return (round(self.plotter.width / 2 + x * self.plotter.scale),
                round(self.plotter.height / 2 - y * self.plotter.scale))

    def pick(self, x, y):
        if not self.packet or not self.packet.get('board_detected'):
            return None
        for index, ball in enumerate(self.packet.get('balls', [])):
            if math.hypot(x - ball['x'], y - ball['y']) <= max(self.radius, 6 / self.plotter.scale):
                return dict(kind='ball', id=index, x=ball['x'], y=ball['y'], rotate=False)
        choices = []
        for car in self.packet.get('markers', []):
            theta = math.radians(car['angle'])
            dx, dy = x - car['x'], y - car['y']
            local = (math.cos(theta) * dx + math.sin(theta) * dy,
                     -math.sin(theta) * dx + math.cos(theta) * dy)
            inside = all((b[0] - a[0]) * (local[1] - a[1]) -
                         (b[1] - a[1]) * (local[0] - a[0]) >= -1e-9
                         for a, b in zip(self.body, self.body[1:] + self.body[:1]))
            if inside:
                choices.append((math.hypot(dx, dy), car))
        if not choices:
            return None
        distance, car = min(choices, key=lambda item: item[0])
        return dict(kind='car', id=car['id'], x=car['x'], y=car['y'], angle=car['angle'],
                    rotate=distance > self.center_radius)

    def move(self, x, y):
        active = self.active
        message = dict(operation='move', kind=active['kind'], id=active['id'])
        if active['rotate']:
            dx, dy = x - active['x'], y - active['y']
            if math.hypot(dx, dy) < 1e-6:
                return
            angle = math.degrees(math.atan2(dy, dx))
            delta = (angle - active['pointer_angle'] + 180) % 360 - 180
            message.update(x=active['x'], y=active['y'], angle=active['angle'] + delta)
        else:
            message.update(x=x - active['offset_x'], y=y - active['offset_y'])
            if active['kind'] == 'car':
                message['angle'] = active['angle']
        self.send(message)

    def on_mouse(self, event, px, py, flags, param=None):
        import cv2
        x, y = self.world_point(px, py)
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.active is not None:
                self.send(dict(operation='release', kind=self.active['kind'], id=self.active['id']))
            self.active = self.pick(x, y)
            if self.active is not None:
                self.active.update(offset_x=x - self.active['x'], offset_y=y - self.active['y'],
                                   pointer_angle=math.degrees(math.atan2(y - self.active['y'], x - self.active['x'])))
                self.send(dict(operation='grab', kind=self.active['kind'], id=self.active['id']))
        elif self.active is not None and event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP):
            self.move(x, y)
            if event == cv2.EVENT_LBUTTONUP or not flags & cv2.EVENT_FLAG_LBUTTON:
                self.send(dict(operation='release', kind=self.active['kind'], id=self.active['id']))
                self.active = None

    def render(self, packet):
        import cv2
        import numpy as np
        self.packet = packet
        frame = self.plotter.render_packet(packet)
        if packet.get('board_detected'):
            for car in packet.get('markers', []):
                theta = math.radians(car['angle'])
                c, s = math.cos(theta), math.sin(theta)
                points = np.array([self.pixels(car['x'] + c * x - s * y, car['y'] + s * x + c * y)
                                   for x, y in self.body], dtype=np.int32)
                selected = self.active and self.active['kind'] == 'car' and self.active['id'] == car['id']
                color = (0, 130, 240) if selected else (160, 110, 30)
                cv2.polylines(frame, [points], True, color, 2, cv2.LINE_AA)
                cv2.circle(frame, self.pixels(car['x'], car['y']), max(3, round(self.center_radius * self.plotter.scale)),
                           color, 1, cv2.LINE_AA)
        cv2.putText(frame, 'Centre: move | Edge: rotate | Ball: move',
                    (8, max(6, min(14, round(self.plotter.scale * .03)))), cv2.FONT_HERSHEY_SIMPLEX,
                    min(.4, frame.shape[1] / 1400), (70, 70, 70), 1, cv2.LINE_AA)
        return frame
