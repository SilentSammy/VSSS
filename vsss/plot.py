"""Persistent field viewer using the existing ArUco BoardPlotter2D renderer."""
import argparse
from copy import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
from threading import Event, Lock, Thread
import time

from .models import WorldState
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
        style = {name: entry[name] for name in ('color', 'text', 'scale', 'image') if name in entry}
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


def render_state(plotter, state):
    return plotter.render_packet(dict(board_detected=state.valid,
        markers=[dict(id=car.id, **asdict(car.pose)) for car in state.cars],
        balls=[dict(id=ball.id, **asdict(ball.pose)) for ball in state.balls]))


class ProviderFeed:
    """Read-only subscriber that follows discovery changes and holds one latest frame."""
    def __init__(self, endpoint=None):
        self.path = Path(endpoint) if endpoint else default_endpoint()
        self.stop = Event()
        self.lock = Lock()
        self.state = WorldState(valid=False)
        self.status = 'Waiting for provider'
        self.source_id = None
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def snapshot(self):
        with self.lock:
            return self.state, self.status, self.source_id

    def _receive(self, event):
        if event.topic == 'gps':
            with self.lock:
                self.state = event.value
        elif event.topic == 'lifecycle' and isinstance(event.value, dict):
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
                        self.status = description['mode']
                    with client.subscribe(['gps', 'lifecycle'], self._receive) as subscription:
                        while not self.stop.wait(.1):
                            if subscription.error:
                                raise ConnectionError(str(subscription.error))
                            if self.path.read_bytes() != discovery:
                                break
            except Exception as error:
                with self.lock:
                    self.status = f'Waiting for provider: {error}'
                    self.source_id = None
            self.stop.wait(.2)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=4)


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
            width, height = setup.board.get_board_dimensions()
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
    window = 'VSSS field'
    latest = None
    duration = args.duration or (1 if args.output else None)
    deadline = time.monotonic() + duration if duration else None
    try:
        if not args.output:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while deadline is None or time.monotonic() < deadline:
            state, status, source = feed.snapshot()
            frame = render_state(plotter, state)
            if source is not None and state.valid:
                latest = frame.copy()
            label = f'{args.setup} | {status}'
            if source is not None and not (
                math.isclose(state.field.width, plotter.board_width, abs_tol=.001) and
                math.isclose(state.field.height, plotter.board_height, abs_tol=.001)):
                label += f' | field mismatch: feed {state.field.width:g}x{state.field.height:g} m'
            cv2.putText(frame, label[:140], (12, 22), cv2.FONT_HERSHEY_SIMPLEX, .45,
                        (60, 60, 60), 1, cv2.LINE_AA)
            if args.output:
                feed.stop.wait(.02)
            else:
                cv2.imshow(window, frame)
                if cv2.waitKey(20) & 0xff in (ord('q'), 27):
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
