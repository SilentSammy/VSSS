"""Host-side BLE UUIDs, payload layouts, and readable robot status values."""
from dataclasses import dataclass
import math
import struct


VELOCITY_UUID = "12345678-1234-5678-1234-56789abcdef1"
LEASE_UUID = "12345678-1234-5678-1234-56789abcdef2"
TRIM_MATRIX_UUID = "12345678-1234-5678-1234-56789abcdef3"
WAYPOINT_UUID = "12345678-1234-5678-1234-56789abcdef6"
CONTROL_UUID = "12345678-1234-5678-1234-56789abcdef7"
CONTROL_RECONNECT_TELEMETRY = 2
CONTROL_RESTART = 3
CONTROLLER_UUID = "12345678-1234-5678-1234-56789abcdef8"
STATUS_UUID = "12345678-1234-5678-1234-56789abcdef9"

TRIM_STRUCT = struct.Struct("<9f")
WAYPOINT_STRUCT = struct.Struct("<fff")
CONTROLLER_STRUCT = struct.Struct("<11f")
STATUS_STRUCT = struct.Struct("<BBBBfffI")

CONTROLLER_KEYS = (
    "kp_dist", "kp_heading", "max_speed", "max_w", "horizon",
    "velocity_alpha", "min_speed", "min_w", "position_tolerance",
    "heading_tolerance_deg", "hold_ms",
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
    return {key: merged[key] for key in CONTROLLER_KEYS}


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
