"""Host-side BLE UUIDs, payload layouts, and readable robot status values."""
from dataclasses import dataclass
import math
import struct
import json


VELOCITY_UUID = "12345678-1234-5678-1234-56789abcdef1"
LEASE_UUID = "12345678-1234-5678-1234-56789abcdef2"
TRIM_MATRIX_UUID = "12345678-1234-5678-1234-56789abcdef3"
WAYPOINT_UUID = "12345678-1234-5678-1234-56789abcdef6"
CONTROL_UUID = "12345678-1234-5678-1234-56789abcdef7"
CONTROL_RECONNECT_TELEMETRY = 2
CONTROL_RESTART = 3
CONTROLLER_UUID = "12345678-1234-5678-1234-56789abcdef8"
STATUS_UUID = "12345678-1234-5678-1234-56789abcdef9"
CONFIG_UUID = "12345678-1234-5678-1234-56789abcdefa"
CONFIG_CHUNK_BYTES = 180
CONFIG_MAX_BYTES = 2048

TRIM_STRUCT = struct.Struct("<9f")
WAYPOINT_STRUCT = struct.Struct("<fff")
CONTROLLER_STRUCT = struct.Struct("<15f")
STATUS_STRUCT = struct.Struct("<BBBBfffI")

CONTROLLER_KEYS = (
    "kp_dist", "kp_heading", "max_speed", "max_w", "horizon",
    "velocity_alpha", "min_speed", "min_w", "position_tolerance",
    "heading_tolerance_deg", "hold_ms", "position_hysteresis",
    "heading_hysteresis", "telemetry_recovery_ms", "waypoint_timeout_ms",
)

MODE_NAMES = ("idle", "manual", "position", "reached", "fault")
FAULT_NAMES = ("none", "telemetry", "timeout")


def normalize_controller(parameters, base=None):
    """Return a complete, firmware-valid controller mapping.

    Partial updates are accepted when a previously read configuration is supplied.
    """
    if not isinstance(parameters, dict):
        raise ValueError("controller parameters must be a mapping")
    unknown = set(parameters) - set(CONTROLLER_KEYS)
    if unknown:
        raise ValueError("unknown controller parameter: " + sorted(unknown)[0])
    merged = {} if base is None else dict(base)
    try:
        merged.update((key, float(value)) for key, value in parameters.items())
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("controller parameters must be finite numbers") from error
    missing = [key for key in CONTROLLER_KEYS if key not in merged]
    if missing:
        raise ValueError("missing controller parameter: " + missing[0])
    if not all(math.isfinite(merged[key]) and merged[key] > 0 for key in CONTROLLER_KEYS):
        raise ValueError("controller parameters must be positive and finite")
    if merged["max_speed"] > 1 or merged["max_w"] > 1:
        raise ValueError("controller output limits must be at most 1")
    if merged["velocity_alpha"] > 1:
        raise ValueError("velocity_alpha must be in (0, 1]")
    if merged["position_hysteresis"] < 1 or merged["heading_hysteresis"] < 1:
        raise ValueError("arrival hysteresis must be at least 1")
    if not 100 <= merged["telemetry_recovery_ms"] <= 10000:
        raise ValueError("telemetry_recovery_ms must be 100 to 10000")
    if not 1000 <= merged["waypoint_timeout_ms"] <= 300000:
        raise ValueError("waypoint_timeout_ms must be 1000 to 300000")
    return {key: merged[key] for key in CONTROLLER_KEYS}


def normalize_robot_config(candidate):
    """Validate a complete remote robot_config.json without exposing secrets."""
    if not isinstance(candidate, dict):
        raise ValueError("robot configuration must be a mapping")
    candidate = dict(candidate)
    candidate.setdefault("telemetry_stale_ms", 800)
    required = {"version", "car_id", "wifi_ssid", "wifi_password", "telemetry_host",
                "telemetry_port", "telemetry_stale_ms", "motor_pins", "trim_matrix", "controller"}
    missing = required - set(candidate)
    if missing:
        raise ValueError("robot configuration is missing " + sorted(missing)[0])
    if candidate["version"] != 1 or type(candidate["car_id"]) is not int or candidate["car_id"] < 0:
        raise ValueError("invalid robot identity")
    if not all(isinstance(candidate[key], str) for key in ("wifi_ssid", "wifi_password", "telemetry_host")):
        raise ValueError("invalid Wi-Fi or telemetry host")
    if not candidate["telemetry_host"] or type(candidate["telemetry_port"]) is not int or not 1 <= candidate["telemetry_port"] <= 65534:
        raise ValueError("invalid telemetry port")
    if (type(candidate["telemetry_stale_ms"]) is not int or
            not 200 <= candidate["telemetry_stale_ms"] <= 5000):
        raise ValueError("telemetry_stale_ms must be 200 to 5000")
    try:
        pins = tuple(tuple(pair) for pair in candidate["motor_pins"])
    except TypeError as error:
        raise ValueError("invalid motor pin mapping") from error
    if (len(pins) != 4 or any(len(pair) != 2 for pair in pins) or
            any(type(pin) is not int or not 0 <= pin <= 21 for pair in pins for pin in pair) or
            len({pin for pair in pins for pin in pair}) != 8):
        raise ValueError("invalid motor pin mapping")
    try:
        gains = tuple(float(value) for value in candidate.get("wheel_gains", (1.,) * 4))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("invalid wheel gains") from error
    if len(gains) != 4 or not all(math.isfinite(value) and .5 <= value <= 1.5 for value in gains):
        raise ValueError("wheel gains must contain four values from 0.5 to 1.5")
    try:
        reverse_gains = tuple(float(value) for value in candidate.get("wheel_reverse_gains", (1.,) * 4))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("invalid reverse wheel gains") from error
    if (len(reverse_gains) != 4 or
            not all(math.isfinite(value) and .5 <= value <= 1.5 for value in reverse_gains)):
        raise ValueError("reverse wheel gains must contain four values from 0.5 to 1.5")
    from ._ble_client import normalize_trim_matrix
    trim = normalize_trim_matrix([candidate["trim_matrix"][i:i + 3] for i in (0, 3, 6)]
                                 if len(candidate["trim_matrix"]) == 9 else candidate["trim_matrix"])
    controller = normalize_controller(candidate["controller"])
    normalized = dict(candidate)
    normalized["motor_pins"] = [list(pair) for pair in pins]
    normalized["wheel_gains"] = list(gains)
    normalized["wheel_reverse_gains"] = list(reverse_gains)
    normalized["trim_matrix"] = [value for row in trim for value in row]
    normalized["controller"] = controller
    try:
        mixed = tuple(float(value) for value in candidate.get("mixed_trim", (0.,) * 18))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("invalid mixed-axis trim") from error
    if len(mixed) != 18 or not all(math.isfinite(v) and -1 <= v <= 1 for v in mixed):
        raise ValueError("mixed_trim needs 18 finite coefficients in [-1, 1]")
    normalized["mixed_trim"] = list(mixed)
    try:
        encoded = json.dumps(normalized, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("robot configuration cannot be encoded") from error
    if len(encoded) > CONFIG_MAX_BYTES:
        raise ValueError("robot configuration exceeds BLE limit")
    return normalized


@dataclass(frozen=True)
class RobotStatus:
    mode: int
    flags: int
    fault: int
    reserved: int
    x: float
    y: float
    angle: float
    packet_count: int

    @property
    def mode_name(self):
        return MODE_NAMES[self.mode] if 0 <= self.mode < len(MODE_NAMES) else "unknown"

    @property
    def fault_name(self):
        return FAULT_NAMES[self.fault] if 0 <= self.fault < len(FAULT_NAMES) else "unknown"

    @property
    def wifi_connected(self):
        return bool(self.flags & 1)

    @property
    def pose_fresh(self):
        return bool(self.flags & 2)
