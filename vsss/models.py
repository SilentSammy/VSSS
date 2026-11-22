"""Transport-independent values shared by providers, clients and recordings."""
from dataclasses import dataclass, field as dc_field, fields, is_dataclass
import math

_records = {}


def record(cls):
    """Opt a dataclass into the shared codec without per-class converters."""
    cls = dataclass(frozen=True)(cls)
    _records[cls.__name__] = cls
    return cls


def encode(value):
    if is_dataclass(value) and not isinstance(value, type):
        if type(value) not in _records.values():
            raise TypeError(f"unregistered record: {type(value).__name__}")
        return {"$record": type(value).__name__, **{f.name: encode(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("wire dictionaries require string keys")
        if "$record" in value:
            raise ValueError("$record is reserved by the record codec")
        return {key: encode(item) for key, item in value.items()}
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError(f"unsupported wire value: {type(value).__name__}")


def decode(value):
    if isinstance(value, list):
        return [decode(item) for item in value]
    if isinstance(value, dict):
        if "$record" in value:
            name = value["$record"]
            if name not in _records:
                raise ValueError(f"unknown record: {name}")
            return _records[name](**{key: decode(item) for key, item in value.items() if key != "$record"})
        return {key: decode(item) for key, item in value.items()}
    return value


def finite(*values):
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("expected finite numbers")


@record
class Pose:
    x: float = 0.
    y: float = 0.
    angle: float = 0.  # degrees, counterclockwise from +x

    def __post_init__(self):
        finite(self.x, self.y, self.angle)


@record
class Field:
    width: float = 1.5
    height: float = 2.9

    def __post_init__(self):
        finite(self.width, self.height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("field dimensions must be positive")


@record
class CarState:
    id: int
    pose: Pose = dc_field(default_factory=Pose)

    def __post_init__(self):
        if type(self.id) is not int or self.id < 0 or not isinstance(self.pose, Pose):
            raise ValueError("invalid car state")


@record
class BallState:
    id: int
    pose: Pose = dc_field(default_factory=Pose)

    def __post_init__(self):
        if type(self.id) is not int or self.id < 0 or not isinstance(self.pose, Pose):
            raise ValueError("invalid ball state")


@record
class WorldState:
    field: Field = dc_field(default_factory=Field)
    cars: list = dc_field(default_factory=list)
    balls: list = dc_field(default_factory=list)
    captured_at: float = 0.  # Unix seconds, for recorded source age
    sequence: int = 0
    valid: bool = True
    frame_age_ms: float = 0.
    source_age: float = 0.  # lower-bound age at delivery; independent of host clock skew
    received_at: float = 0.  # delivery wall time; captured_at preserves source time

    def __post_init__(self):
        if not isinstance(self.field, Field):
            raise ValueError("field must be a Field")
        for items, kind in ((self.cars, CarState), (self.balls, BallState)):
            if not isinstance(items, list) or any(not isinstance(item, kind) for item in items):
                raise ValueError("invalid world objects")
            if len({item.id for item in items}) != len(items):
                raise ValueError("duplicate object IDs")
        finite(self.captured_at)
        finite(self.frame_age_ms, self.source_age, self.received_at)
        if self.frame_age_ms < 0 or self.source_age < 0:
            raise ValueError('negative feedback age')
        if type(self.sequence) is not int or self.sequence < 0 or type(self.valid) is not bool:
            raise ValueError("invalid world timing/validity")


@record
class Velocity:
    x: float = 0.
    y: float = 0.
    w: float = 0.  # normalized inputs, all in [-1, 1]

    def __post_init__(self):
        finite(self.x, self.y, self.w)
        if any(abs(v) > 1 for v in (self.x, self.y, self.w)):
            raise ValueError("velocity components must be in [-1, 1]")


@record
class CommandEvent:
    car_id: int
    operation: str
    value: object
    session: str
    reason: str = "command"

    def __post_init__(self):
        if type(self.car_id) is not int or self.car_id < 0:
            raise ValueError("invalid command event")
        expected = {'velocity': Velocity, 'waypoint': Pose}.get(self.operation)
        if expected is None or not isinstance(self.value, expected) or not isinstance(self.session, str) or not isinstance(self.reason, str):
            raise ValueError("invalid command metadata")


@record
class Event:
    topic: str
    value: object
    sequence: int
    timestamp: float  # provider monotonic seconds

    def __post_init__(self):
        finite(self.timestamp)
        if self.topic not in {"gps", "commands", "lifecycle"} or type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("invalid event metadata")


@record
class StatePatch:
    """Only listed poses change. Edits stop edited cars; others keep moving."""
    cars: list = dc_field(default_factory=list)
    balls: list = dc_field(default_factory=list)


@record
class BodyMotion:
    """World-frame m/s and degrees/s, not normalized motor commands."""
    kind: str
    id: int
    vx: float = 0.
    vy: float = 0.
    omega: float = 0.

    def __post_init__(self):
        finite(self.vx, self.vy, self.omega)
        if self.kind not in ('car', 'ball') or type(self.id) is not int or self.id < 0:
            raise ValueError('invalid body motion')
        if math.hypot(self.vx, self.vy) > 5 or abs(self.omega) > 1080:
            raise ValueError('initial motion exceeds simulation bounds')


@record
class SimScenario:
    state: WorldState
    motion: list = dc_field(default_factory=list)
    config: dict = dc_field(default_factory=dict)
    seed: object = None
    wait_for_control: bool = False

    def __post_init__(self):
        if not isinstance(self.state, WorldState) or not isinstance(self.config, dict):
            raise ValueError('invalid scenario state/config')
        if not isinstance(self.motion, list) or any(not isinstance(m, BodyMotion) for m in self.motion):
            raise ValueError('invalid initial motion')
        keys = [(m.kind, m.id) for m in self.motion]
        valid = {('car', c.id) for c in self.state.cars} | {('ball', b.id) for b in self.state.balls}
        if len(set(keys)) != len(keys) or not set(keys) <= valid:
            raise ValueError('duplicate or unknown motion identity')
        if self.seed is not None and type(self.seed) is not int or type(self.wait_for_control) is not bool:
            raise ValueError('invalid scenario seed/start gate')
