"""Persistent field viewer using the existing ArUco BoardPlotter2D renderer."""
import argparse
from dataclasses import asdict
import math
from pathlib import Path
from threading import Event, Lock, Thread
import time

from .models import WorldState
from .provider import VSSSProvider
from .rpc import default_endpoint


def make_plotter(field, width=720):
    import cv2
    from aruco_gps.board_config import GridboardConfig
    from aruco_gps.plotter2d import BoardPlotter2D
    # Same 3x5 printed board artwork as setup_160cm; no camera/HTTP dependency.
    board = GridboardConfig(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50),
                            size=(3, 5), marker_length=.1, board_width=1.5, print_width=1.6)
    setup = dict(board=board.to_dict(), marker_size_m=.09, plot_margin_m=.05,
                 ball_detector=dict(radius_m=.02))
    # Keep alternate scenario dimensions usable while retaining the board artwork.
    setup['board']['dimensions_m'] = [field.width, field.height]
    setup['board']['print_dimensions_m'] = [field.width + .1, field.height + .1]
    return BoardPlotter2D.from_metadata(setup, width_px=width)


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint')
    parser.add_argument('--width', type=int, default=720)
    parser.add_argument('--duration', type=float)
    parser.add_argument('--output', type=Path, help='Save the final frame without opening a window')
    parser.add_argument('--persist', action='store_true', help='Accepted for convenience; reconnecting is now the default')
    args = parser.parse_args(argv)
    if args.width < 100:
        parser.error('--width must be at least 100')
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error('--duration must be positive and finite')
    if args.output and args.output.exists():
        parser.error('--output must be a new file')
    import cv2
    feed = ProviderFeed(args.endpoint)
    window = 'VSSS field'
    plotter = field = None
    latest = None
    duration = args.duration or (1 if args.output else None)
    deadline = time.monotonic() + duration if duration else None
    try:
        if not args.output:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while deadline is None or time.monotonic() < deadline:
            state, status, source = feed.snapshot()
            if field != state.field:
                field = state.field
                plotter = make_plotter(field, args.width)
            frame = render_state(plotter, state)
            if source is not None and state.valid:
                latest = frame.copy()
            cv2.putText(frame, status[:95], (12, 22), cv2.FONT_HERSHEY_SIMPLEX, .45,
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
