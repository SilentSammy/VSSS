"""Persistent field viewer using the existing ArUco BoardPlotter2D renderer."""
import argparse
from collections import deque
from copy import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
from threading import Event, Lock, Thread
import time

from .models import CommandEvent, WorldState
from .provider import VSSSProvider
from .rpc import default_endpoint


def lab_setups():
    from aruco_gps.vision_setup import DEFAULT_SETUP, VISION_SETUPS
    default = next(name for name, setup in VISION_SETUPS.items() if setup is DEFAULT_SETUP)
    return VISION_SETUPS, default


def marker_styles(path):
    """Read local visual overrides; height calibration belongs to detection."""
    import cv2
    import numpy as np
    if path is None:
        return {}
    path = Path(path).resolve()
    with path.open(encoding='utf-8-sig') as file:
        entries = json.load(file)
    if not isinstance(entries, dict):
        raise ValueError('marker config must be an object keyed by marker ID')
    styles = {}
    for key, entry in entries.items():
        marker_id = int(key)
        if marker_id < 0 or not isinstance(entry, dict):
            raise ValueError(f'invalid marker config entry: {key}')
        style = {name: entry[name] for name in ('color', 'text', 'scale', 'image', 'size_m', 'stretch') if name in entry}
        color = style.get('color')
        if 'color' in style and (not isinstance(color, list) or len(color) != 3 or
                                  any(type(c) is not int or not 0 <= c <= 255 for c in color)):
            raise ValueError(f'marker {key}: color must be three RGB integers in 0..255')
        if 'text' in style and (not isinstance(style['text'], str) or
                               any(c != '\n' and not ' ' <= c <= '~' for c in style['text'])):
            raise ValueError(f'marker {key}: text must be printable ASCII')
        if 'scale' in style and (type(style['scale']) not in (int, float) or
                                 not math.isfinite(style['scale']) or style['scale'] <= 0):
            raise ValueError(f'marker {key}: scale must be positive and finite')
        if 'image' in style:
            image_path = Path(style['image'])
            if not image_path.is_absolute():
                image_path = path.parent / image_path
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if image is None or not (image.ndim == 2 or image.ndim == 3 and image.shape[2] in (3, 4)):
                raise ValueError(f'marker {key}: unsupported image {image_path}')
            style['image'] = image
        if 'size_m' in style and (type(style['size_m']) not in (int, float) or
                                 not math.isfinite(style['size_m']) or style['size_m'] <= 0):
            raise ValueError(f'marker {key}: size_m must be positive and finite')
        if 'stretch' in style and type(style['stretch']) is not bool:
            raise ValueError(f'marker {key}: stretch must be boolean')
        styles[marker_id] = style
    return styles


def make_plotter(setup=None, width=720, marker_config=None):
    from aruco_gps import vision_setup
    from aruco_gps.plotter2d import BoardPlotter2D
    setup = vision_setup.VISION_SETUPS[setup] if isinstance(setup, str) else setup or vision_setup.DEFAULT_SETUP
    board = copy(setup.board)
    # Board paths in vision_setup are relative to the vision repository, not cwd.
    if not Path(board.filename).is_absolute():
        board.filename = str(Path(vision_setup.__file__).parent / board.filename)
    plotter = BoardPlotter2D(board, setup.detector.getDictionary(),
                             margin=setup.plot_margin_m, width_px=width)
    plotter.marker_overrides = {key: dict(style) for key, style in plotter.marker_overrides.items()}
    for key, style in marker_styles(marker_config if marker_config is not None else setup.marker_config_path).items():
        plotter.marker_overrides.setdefault(key, {}).update(style)
    plotter.marker_size_m = setup.marker_size_m
    plotter.ball_radius_m = setup.ball_detector.radius_m if setup.ball_detector else .02
    return plotter


def render_state(plotter, state, waypoints=None):
    frame = plotter.render_packet(dict(board_detected=state.valid,
        markers=[dict(id=car.id, **asdict(car.pose)) for car in state.cars],
        balls=[dict(id=ball.id, **asdict(ball.pose)) for ball in state.balls]))
    if not waypoints: return frame
    import cv2
    def pixel(x, y):
        return (round(plotter.width/2+x*plotter.scale), round(plotter.height/2-y*plotter.scale))
    cars = {car.id: car.pose for car in state.cars} if state.valid else {}
    for number, target in sorted(waypoints.items()):
        color = [(40, 170, 240), (210, 150, 40), (160, 75, 190), (80, 165, 70)][(number-1) % 4]
        center = pixel(target.x, target.y)
        if number in cars:
            start = pixel(cars[number].x, cars[number].y)
            # A dashed guide leaves the field and car artwork visible.
            length = math.dist(start, center)
            for distance in range(0, round(length), 12):
                a, b = distance/max(1,length), min(distance+6,length)/max(1,length)
                cv2.line(frame, tuple(round(p+(q-p)*a) for p,q in zip(start,center)),
                         tuple(round(p+(q-p)*b) for p,q in zip(start,center)), color, 1, cv2.LINE_AA)
        cv2.circle(frame, center, max(5, round(.045*plotter.scale)), color, 2, cv2.LINE_AA)
        heading = math.radians(target.angle)
        tip = pixel(target.x+.13*math.cos(heading), target.y+.13*math.sin(heading))
        cv2.arrowedLine(frame, center, tip, color, 2, cv2.LINE_AA, tipLength=.3)
        cv2.putText(frame, f'car{number} target', (center[0]+8,center[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1, cv2.LINE_AA)
    return frame


def is_close_key(key):
    """Escape closes the plot; teleop keys must pass through globally."""
    return key & 0xff == 27


class ProviderFeed:
    """Read-only subscriber that follows discovery changes and holds one latest frame."""
    def __init__(self, endpoint=None):
        self.path = Path(endpoint) if endpoint else default_endpoint()
        self.stop = Event()
        self.lock = Lock()
        self.state = WorldState(valid=False)
        self.status = 'Waiting for provider'
        self.source_id = None
        self.description = {}
        self._waypoints = {}
        self._seen_commands = set()
        self._waypoint_epoch = 0
        self._edits = deque()
        self._metadata_dirty = False
        self._last_event = time.monotonic()
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def snapshot(self):
        with self.lock:
            return self.state, self.status, self.source_id

    def info(self):
        with self.lock: return self.description

    def waypoints(self):
        with self.lock: return dict(self._waypoints)

    def _command(self, command):
        """Caller holds the feed lock. Never sends a command to a car."""
        if not isinstance(command, CommandEvent): return
        self._seen_commands.add(command.car_id)
        if command.operation == 'waypoint': self._waypoints[command.car_id] = command.value
        else: self._waypoints.pop(command.car_id, None)

    def _prune_reached(self):
        if not self.state.valid: return
        for car in self.state.cars:
            target = self._waypoints.get(car.id)
            if target and math.hypot(car.pose.x-target.x,car.pose.y-target.y) <= .025 and abs(
                    (car.pose.angle-target.angle+180)%360-180) <= 5:
                self._waypoints.pop(car.id, None)

    def gesture(self, operation, kind, number, pose=None, *, source_id):
        with self.lock:
            if source_id != self.source_id or not self.description.get('capabilities', {}).get('simulation_edit'):
                return
            item = (operation, kind, number, pose, source_id)
            # Only adjacent moves coalesce; grab/release ordering is retained.
            if operation == 'move' and self._edits and self._edits[-1][0] == 'move':
                self._edits[-1] = item
            else:
                self._edits.append(item)

    def _edit_pending(self, client, grab):
        while True:
            with self.lock:
                if not self._edits: return grab
                operation, kind, number, pose, source = self._edits.popleft()
            if source != self.source_id: continue
            try:
                if operation == 'grab':
                    result = client.edit('grab', kind, number, source_id=source)
                    grab = (kind, number, result['token'])
                elif grab and grab[:2] == (kind, number):
                    client.edit(operation, kind, number, pose, source_id=source, token=grab[2])
                    if operation == 'release': grab = None
            except (ValueError, RuntimeError) as error:
                with self.lock: self.status = f'Edit rejected: {error}'
                if grab:
                    try: client.edit('release', grab[0], grab[1], source_id=source, token=grab[2])
                    except (ValueError, RuntimeError): pass
                grab = None

    def _receive(self, event):
        if event.topic == 'gps':
            with self.lock:
                self.state = event.value
                self._last_event = time.monotonic()
                self._prune_reached()
        elif event.topic == 'commands':
            with self.lock:
                self._command(event.value)
                self._prune_reached()
        elif event.topic == 'lifecycle' and isinstance(event.value, dict):
            if event.value.get('kind') == 'scenario_loaded':
                with self.lock:
                    self._metadata_dirty = True
                    self._waypoints.clear()
                    self._waypoint_epoch += 1
            if event.value.get('kind') in ('state_edited', 'simulation_edit'):
                patch = event.value.get('patch')
                ids = [car.id for car in patch.cars] if patch is not None else []
                if event.value.get('object_kind') == 'car': ids.append(event.value['id'])
                with self.lock:
                    for number in ids:
                        self._waypoints.pop(number, None)
                        self._seen_commands.add(number)
            if event.value.get('kind') == 'playback_end':
                with self.lock:
                    self.status = 'Playback finished' if event.value.get('complete') else str(event.value)

    def _run(self):
        while not self.stop.is_set():
            try:
                discovery = self.path.read_bytes()
                with VSSSProvider.connect(self.path, timeout=1) as client:
                    description = client.describe()
                    state = client.gps.state()
                    with self.lock:
                        self.state, self.source_id = state, description['source_id']
                        self.description = description
                        self._waypoints.clear()
                        self._seen_commands.clear()
                        self._waypoint_epoch += 1
                        self._edits.clear()
                        self.status = description['mode']
                    grab = None
                    last_poll = time.monotonic()
                    with client.subscribe(['gps', 'commands', 'lifecycle'], self._receive) as subscription:
                        # Subscribe first, then seed commands sent before the plot
                        # opened. Events received meanwhile win over the snapshot.
                        with self.lock: epoch = self._waypoint_epoch
                        commands = client.cars.observed()
                        with self.lock:
                            if epoch == self._waypoint_epoch:
                                for command in commands:
                                    if command.car_id not in self._seen_commands: self._command(command)
                                self._prune_reached()
                        while not self.stop.wait(.02):
                            if subscription.error:
                                raise ConnectionError(str(subscription.error))
                            if self.path.read_bytes() != discovery:
                                break
                            with self.lock: dirty = self._metadata_dirty
                            if dirty:
                                info = client.describe()
                                with self.lock:
                                    self.description, self._metadata_dirty = info, False
                            if description['mode'] != 'playback' and time.monotonic()-last_poll > .2:
                                last_poll = time.monotonic()
                                with self.lock: silent = last_poll-self._last_event > .2
                                if silent:
                                    state = client.gps.state()
                                    with self.lock:
                                        self.state = state
                                        self._prune_reached()
                            grab = self._edit_pending(client, grab)
            except Exception as error:
                with self.lock:
                    self.status = f'Waiting for provider: {error}'
                    self.source_id = None
                    self.description = {}
                    self._waypoints.clear()
                    self._edits.clear()
            self.stop.wait(.2)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=4)


class SimDrag:
    """Mouse geometry only; the feed worker owns all RPC and grab lifetimes."""
    def __init__(self, plotter, feed):
        self.plotter, self.feed, self.active = plotter, feed, None

    def point(self, px, py):
        return ((px-self.plotter.width/2)/self.plotter.scale,
                -(py-self.plotter.height/2)/self.plotter.scale)

    def mouse(self, event, px, py, flags, param=None):
        import cv2
        state, _, source = self.feed.snapshot()
        info = self.feed.info()
        if not info.get('capabilities', {}).get('simulation_edit') or self.active and (
            self.active['source'] != source or self.active['generation'] != info['simulation']['generation']):
            self.active = None
            return
        x, y = self.point(px, py)
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.active:
                self.feed.gesture('release', self.active['kind'], self.active['number'], source_id=source)
                self.active = None
            radius = info['simulation']['ball_radius']
            body = info['simulation']['body']
            picked = None
            for ball in state.balls:
                if math.hypot(x-ball.pose.x, y-ball.pose.y) <= radius:
                    picked = ('ball', ball.id, ball.pose, False)
                    break
            if picked is None:
                for car in state.cars:
                    theta = math.radians(car.pose.angle)
                    dx, dy = x-car.pose.x, y-car.pose.y
                    local = (math.cos(theta)*dx+math.sin(theta)*dy, -math.sin(theta)*dx+math.cos(theta)*dy)
                    if all((b[0]-a[0])*(local[1]-a[1])-(b[1]-a[1])*(local[0]-a[0]) >= 0
                           for a, b in zip(body, body[1:]+body[:1])):
                        picked = ('car', car.id, car.pose, math.hypot(dx, dy) > .03)
                        break
            if picked:
                kind, number, pose, rotate = picked
                self.active = dict(kind=kind, number=number, pose=pose, rotate=rotate, source=source,
                                   generation=info['simulation']['generation'],
                                   offset=(x-pose.x, y-pose.y), pointer=math.degrees(math.atan2(y-pose.y, x-pose.x)))
                self.feed.gesture('grab', kind, number, source_id=source)
        elif self.active and event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP):
            from .models import Pose
            active = self.active
            pose = active['pose']
            if active['rotate']:
                angle = math.degrees(math.atan2(y-pose.y, x-pose.x))
                delta = (angle-active['pointer']+180) % 360-180
                target = Pose(pose.x, pose.y, pose.angle+delta)
            else:
                target = Pose(x-active['offset'][0], y-active['offset'][1], pose.angle)
            self.feed.gesture('move', active['kind'], active['number'], target, source_id=source)
            if event == cv2.EVENT_LBUTTONUP or not flags & cv2.EVENT_FLAG_LBUTTON:
                self.feed.gesture('release', active['kind'], active['number'], source_id=source)
                self.active = None

    def render(self, frame, state):
        import cv2
        info = self.feed.info().get('simulation')
        if info is None: return frame
        cv2.putText(frame, 'Centre: move | Edge: rotate | Ball: move', (8, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (70, 70, 70), 1, cv2.LINE_AA)
        return frame


class EmbeddedPlot:
    """Window-free plot for other GUIs, with the same renderer and sim gestures.

    Pointer positions use rendered pixels. Each gesture carries the displayed
    source/generation so delayed GUI input cannot edit a replacement world.
    A short renewed lease releases a grab if its GUI disappears.
    """
    def __init__(self, endpoint=None, *, width=480, setup=None):
        self.plotter = make_plotter(setup, width)
        self.feed = ProviderFeed(endpoint)
        self.drag = SimDrag(self.plotter, self.feed)
        self.lock = Lock()
        self.owner = None
        self.touched = 0.

    def _release(self):
        if self.drag.active:
            active = self.drag.active
            self.feed.gesture('release', active['kind'], active['number'], source_id=active['source'])
        self.drag.active, self.owner = None, None

    def frame(self):
        with self.lock:
            if self.owner and time.monotonic()-self.touched > 1:
                self._release()
            state, status, source = self.feed.snapshot()
            info = self.feed.info()
            frame = self.drag.render(render_state(self.plotter, state, self.feed.waypoints()), state)
            metadata = dict(source_id=source, generation=info.get('simulation', {}).get('generation'),
                            editable=info.get('capabilities', {}).get('simulation_edit', False), status=status)
            return frame, metadata

    def pointer(self, operation, x, y, *, owner, source_id, generation):
        import cv2
        if operation not in ('down', 'move', 'up') or not isinstance(owner, str) or not 1 <= len(owner) <= 80:
            raise ValueError('Invalid plot pointer')
        if not all(type(v) in (int, float) and math.isfinite(v) for v in (x, y)):
            raise ValueError('Invalid plot coordinates')
        with self.lock:
            if self.owner and time.monotonic()-self.touched > 1: self._release()
            info = self.feed.info()
            if source_id != self.feed.snapshot()[2] or generation != info.get('simulation', {}).get('generation'):
                raise ValueError('Plot changed; wait for a fresh frame')
            if not info.get('capabilities', {}).get('simulation_edit'):
                raise ValueError('Dragging is available in simulation only')
            if self.owner and self.owner != owner:
                raise ValueError('Plot is being dragged in another editor')
            if operation == 'down': self.owner = owner
            elif self.owner != owner: return
            self.touched = time.monotonic()
            event = dict(down=cv2.EVENT_LBUTTONDOWN, move=cv2.EVENT_MOUSEMOVE, up=cv2.EVENT_LBUTTONUP)[operation]
            self.drag.mouse(event, x, y, cv2.EVENT_FLAG_LBUTTON if operation != 'up' else 0)
            if operation == 'up': self.owner = None

    def close(self):
        with self.lock: self._release()
        self.feed.close()


def main(argv=None):
    setups, default = lab_setups()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setup', choices=tuple(setups), default=default,
                        help=f'Local lab setup/artwork (default: {default}); match the lab server manually')
    parser.add_argument('--marker-config', type=Path, help='Local marker JSON; defaults to the selected setup configuration')
    parser.add_argument('--list-setups', action='store_true', help='List local setups and exit')
    parser.add_argument('--endpoint')
    parser.add_argument('--width', type=int, default=720)
    parser.add_argument('--duration', type=float)
    parser.add_argument('--output', type=Path, help='Save the final frame without opening a window')
    parser.add_argument('--persist', action='store_true', help='Accepted for convenience; reconnecting is now the default')
    args = parser.parse_args(argv)
    if args.list_setups:
        for name, setup in setups.items():
            width, height = setup.board.get_print_dimensions()
            print(f'{name}: {width:g} x {height:g} m' + (' (default)' if name == default else ''))
        return
    if args.width < 100:
        parser.error('--width must be at least 100')
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error('--duration must be positive and finite')
    if args.output and args.output.exists():
        parser.error('--output must be a new file')
    import cv2
    try:
        plotter = make_plotter(args.setup, args.width, args.marker_config)
    except (OSError, ValueError, TypeError, cv2.error) as error:
        parser.error(str(error))
    feed = ProviderFeed(args.endpoint)
    drag = SimDrag(plotter, feed)
    window = 'VSSS field'
    latest = None
    duration = args.duration or (1 if args.output else None)
    deadline = time.monotonic() + duration if duration else None
    try:
        if not args.output:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window, drag.mouse)
        while deadline is None or time.monotonic() < deadline:
            state, status, source = feed.snapshot()
            frame = render_state(plotter, state, feed.waypoints())
            if not args.output: frame = drag.render(frame, state)
            if source is not None and state.valid:
                latest = frame.copy()
            label = f'{args.setup} | {status}'
            if source is not None and not (
                math.isclose(state.field.width, plotter.print_width, abs_tol=.001) and
                math.isclose(state.field.height, plotter.print_height, abs_tol=.001)):
                label += f' | field mismatch: feed {state.field.width:g}x{state.field.height:g} m'
            cv2.putText(frame, label[:140], (12, 22), cv2.FONT_HERSHEY_SIMPLEX, .45,
                        (60, 60, 60), 1, cv2.LINE_AA)
            if args.output:
                feed.stop.wait(.02)
            else:
                cv2.imshow(window, frame)
                if is_close_key(cv2.waitKey(20)):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        feed.close()
        if not args.output:
            cv2.destroyAllWindows()
    if args.output:
        if latest is None:
            raise RuntimeError('provider produced no valid frames')
        if not cv2.imwrite(str(args.output), latest):
            raise OSError(f'Could not save {args.output}')
        print('Saved', args.output)


if __name__ == '__main__':
    main()
