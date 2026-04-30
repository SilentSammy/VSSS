import json
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BallState:
    x: float
    y: float


@dataclass
class PlayerState:
    id: int
    x: float
    y: float
    angle: float


@dataclass
class GameState:
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
    def from_json(s: str) -> 'GameState':
        d = json.loads(s)
        return GameState(
            balls=[BallState(**b) for b in d['balls']],
            players=[PlayerState(**p) for p in d['players']],
            timestamp=d['timestamp'],
        )


@dataclass
class PlotUpdate:
    """All data the plotter needs for one frame: game state plus optional visual overlays.

    Sticky semantics — None means "keep previous value":
      svg_points: None = keep | [] = clear | [[x,y],...] = set new path
      waypoint:   None = keep | () = clear | (x, y) = set new target
    """
    game_state: GameState
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
            game_state=GameState.from_json(json.dumps(d['game_state'])),
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
