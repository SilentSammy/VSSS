import json
import time
from dataclasses import dataclass
from typing import List, Optional
import zmq


@dataclass
class PlotBallState:
    x: float
    y: float


@dataclass
class PlotPlayerState:
    id: int
    x: float
    y: float
    angle: float


@dataclass
class PlotGameState:
    balls: List = None
    players: List = None
    timestamp: float = None

    def __post_init__(self):
        if self.balls is None:
            self.balls = []
        if self.players is None:
            self.players = []

    def to_json(self) -> str:
        return json.dumps({
            'balls':   [{'x': b.x, 'y': b.y} for b in self.balls],
            'players': [{'id': p.id, 'x': p.x, 'y': p.y, 'angle': p.angle} for p in self.players],
            'timestamp': self.timestamp,
        })

    @staticmethod
    def from_json(s: str) -> 'PlotGameState':
        d = json.loads(s)
        return PlotGameState(
            balls=[PlotBallState(**b) for b in d['balls']],
            players=[PlotPlayerState(**p) for p in d['players']],
            timestamp=d['timestamp'],
        )


@dataclass
class PlotUpdate:
    """All data the plotter needs for one frame: game state plus optional visual overlays.

    Sticky semantics — None means "keep previous value":
      svg_points: None = keep | [] = clear | [[x,y],...] = set new path
      waypoint:   None = keep | () = clear | (x, y) = set new target
    """
    game_state: PlotGameState
    svg_points: Optional[List] = None
    waypoint: Optional[tuple] = None

    def to_json(self) -> str:
        return json.dumps({
            'game_state': json.loads(self.game_state.to_json()),
            'svg_points': self.svg_points,
            'waypoint': list(self.waypoint) if self.waypoint is not None else None,
        })

    @staticmethod
    def from_json(s: str) -> 'PlotUpdate':
        d = json.loads(s)
        raw_wp = d.get('waypoint')
        return PlotUpdate(
            game_state=PlotGameState.from_json(json.dumps(d['game_state'])),
            svg_points=d.get('svg_points'),
            waypoint=tuple(raw_wp) if raw_wp is not None else None,
        )


@dataclass
class ClickEvent:
    x: float
    y: float
    timestamp: float

    def to_json(self) -> str:
        return json.dumps({'x': self.x, 'y': self.y, 'timestamp': self.timestamp})

    @staticmethod
    def from_json(s: str) -> 'ClickEvent':
        return ClickEvent(**json.loads(s))


GAME_STATE_PORT = 5556
CLICK_PORT      = 5557


class PlotReceiver:
    """ZMQ helper for the plotter process.

    Subscribes to PlotUpdates from main and publishes ClickEvents back.
    Use as a context manager or call close() when done.
    """

    def __init__(self, game_state_port: int = GAME_STATE_PORT, click_port: int = CLICK_PORT):
        self._ctx = zmq.Context()
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.CONFLATE, 1)  # keep only latest (no backlog)
        self._sub.connect(f"tcp://localhost:{game_state_port}")
        self._sub.setsockopt_string(zmq.SUBSCRIBE, '')
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.bind(f"tcp://*:{click_port}")
        print(f"[PlotReceiver] Subscribed to game state on :{game_state_port}")
        print(f"[PlotReceiver] Publishing clicks on :{click_port}")

    def recv(self) -> 'PlotUpdate | None':
        """Non-blocking receive. Returns a PlotUpdate or None if nothing arrived."""
        try:
            return PlotUpdate.from_json(self._sub.recv_string(zmq.NOBLOCK))
        except zmq.Again:
            return None

    def send_click(self, x: float, y: float):
        """Publish a click event back to main."""
        self._pub.send_string(ClickEvent(x=x, y=y, timestamp=time.time()).to_json())

    def close(self):
        self._sub.close()
        self._pub.close()
        self._ctx.term()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class PlotPublisher:
    """ZMQ helper for demo/main processes.

    Publishes game state to the plotter and receives click events back.
    Accepts raw game_det GameState objects — conversion is handled internally.
    Use as a context manager or call close() when done.
    """

    def __init__(self, game_state_port: int = GAME_STATE_PORT, click_port: int = CLICK_PORT):
        self._ctx = zmq.Context()
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.bind(f"tcp://*:{game_state_port}")
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.connect(f"tcp://localhost:{click_port}")
        self._sub.setsockopt_string(zmq.SUBSCRIBE, '')

    def update(self, game_state, waypoint=None, svg_points=None):
        """Publish game state to the plotter.

        game_state: a game_det.GameState (or any object with .balls/.players/.timestamp)
        waypoint:   None = clear | (x, y) = set new target marker
        svg_points: None = clear | [[x,y],...] = draw path
        """
        plot_gs = PlotGameState(
            balls=[PlotBallState(x=b.x, y=b.y) for b in game_state.balls],
            players=[PlotPlayerState(id=p.id, x=p.x, y=p.y, angle=p.angle) for p in game_state.players],
            timestamp=game_state.timestamp,
        )
        self._pub.send_string(PlotUpdate(
            game_state=plot_gs,
            waypoint=waypoint if waypoint is not None else (),    # None → clear
            svg_points=svg_points if svg_points is not None else [],  # None → clear
        ).to_json())

    def get_clicks(self) -> list:
        """Drain all pending click events. Returns a list of (x, y) tuples."""
        clicks = []
        try:
            while True:
                event = ClickEvent.from_json(self._sub.recv_string(zmq.NOBLOCK))
                clicks.append((event.x, event.y))
        except zmq.Again:
            pass
        return clicks

    def close(self):
        self._pub.close()
        self._sub.close()
        self._ctx.term()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
