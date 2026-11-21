"""Validated, persistent per-robot configuration for MicroPython."""
try:
    import ujson as json
except ImportError:
    import json
import math
import os

DEFAULT_MOTOR_PINS = ((3, 4), (1, 0), (7, 5), (8, 9))

try:
    from onboard_controller import CONTROLLER_KEYS, DEFAULT_CONTROLLER
except ImportError:  # CPython tests import this module as mp.robot_config.
    from .onboard_controller import CONTROLLER_KEYS, DEFAULT_CONTROLLER


DEFAULTS = {
    "version": 1,
    "car_id": 1,
    "wifi_ssid": "embedded",
    "wifi_password": "12345678",
    "telemetry_host": "192.168.12.1",
    "telemetry_port": 5000,
    "telemetry_stale_ms": 800,
    "motor_pins": DEFAULT_MOTOR_PINS,
    "wheel_gains": [1., 1., 1., 1.],
    "wheel_reverse_gains": [1., 1., 1., 1.],
    "trim_matrix": [1., 0., 0., 0., 1., 0., 0., 0., 1.],
    "mixed_trim": [0.] * 18,
    "controller": dict(DEFAULT_CONTROLLER),
}


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def validate(candidate):
    config = dict(DEFAULTS)
    config["controller"] = dict(DEFAULTS["controller"])
    if isinstance(candidate, dict):
        for key in ("version", "car_id", "wifi_ssid", "wifi_password",
                    "telemetry_host", "telemetry_port", "telemetry_stale_ms",
                    "motor_pins", "wheel_gains", "wheel_reverse_gains", "trim_matrix",
                    "mixed_trim"):
            if key in candidate:
                config[key] = candidate[key]
        if isinstance(candidate.get("controller"), dict):
            config["controller"].update(candidate["controller"])
    if config["version"] != 1 or not isinstance(config["car_id"], int) or config["car_id"] < 0:
        raise ValueError("invalid robot identity")
    if not isinstance(config["wifi_ssid"], str) or not isinstance(config["wifi_password"], str):
        raise ValueError("invalid WiFi configuration")
    if (not isinstance(config["telemetry_host"], str) or not config["telemetry_host"] or
            not isinstance(config["telemetry_port"], int) or
            not 1 <= config["telemetry_port"] <= 65534):
        raise ValueError("invalid telemetry configuration")
    if (type(config["telemetry_stale_ms"]) is not int or
            not 200 <= config["telemetry_stale_ms"] <= 5000):
        raise ValueError("telemetry stale threshold is out of range")
    try:
        pins = tuple(tuple(pair) for pair in config["motor_pins"])
    except TypeError:
        raise ValueError("invalid motor pin mapping")
    if (len(pins) != 4 or any(len(pair) != 2 for pair in pins) or
            any(type(pin) is not int or not 0 <= pin <= 21 for pair in pins for pin in pair) or
            len({pin for pair in pins for pin in pair}) != 8):
        raise ValueError("invalid motor pin mapping")
    config["motor_pins"] = [list(pair) for pair in pins]
    try:
        gains = tuple(config["wheel_gains"])
    except TypeError:
        raise ValueError("invalid wheel gains")
    if len(gains) != 4 or not all(_finite(gain) and .5 <= gain <= 1.5 for gain in gains):
        raise ValueError("wheel gains must contain four values from 0.5 to 1.5")
    config["wheel_gains"] = list(gains)
    try:
        reverse_gains = tuple(config["wheel_reverse_gains"])
    except TypeError:
        raise ValueError("invalid reverse wheel gains")
    if len(reverse_gains) != 4 or not all(_finite(gain) and .5 <= gain <= 1.5
                                          for gain in reverse_gains):
        raise ValueError("reverse wheel gains must contain four values from 0.5 to 1.5")
    config["wheel_reverse_gains"] = list(reverse_gains)
    matrix = tuple(config["trim_matrix"])
    if len(matrix) != 9 or not all(_finite(v) and
            (0 if i in (0, 4, 8) else -2) <= v <= 2 for i, v in enumerate(matrix)):
        raise ValueError("invalid trim matrix")
    config["trim_matrix"] = list(matrix)
    try:
        mixed = tuple(config["mixed_trim"])
    except TypeError:
        raise ValueError("invalid mixed-axis trim")
    if len(mixed) != 18 or not all(_finite(v) and -1 <= v <= 1 for v in mixed):
        raise ValueError("mixed_trim needs 18 finite coefficients in [-1, 1]")
    config["mixed_trim"] = list(mixed)
    controller = config["controller"]
    if not all(_finite(controller.get(key)) and controller[key] > 0
               for key in CONTROLLER_KEYS):
        raise ValueError("invalid controller configuration")
    if controller["max_speed"] > 1 or controller["max_w"] > 1 or not 0 < controller["velocity_alpha"] <= 1:
        raise ValueError("controller limits are out of range")
    if controller["position_hysteresis"] < 1 or controller["heading_hysteresis"] < 1:
        raise ValueError("controller hysteresis is out of range")
    if not 100 <= controller["telemetry_recovery_ms"] <= 10000:
        raise ValueError("telemetry recovery is out of range")
    if not 1000 <= controller["waypoint_timeout_ms"] <= 300000:
        raise ValueError("waypoint timeout is out of range")
    return config


class RobotConfig:
    def __init__(self, path="robot_config.json"):
        self.path = path
        self.data = validate(self._read())

    def _read(self):
        try:
            with open(self.path, "r") as source:
                return json.load(source)
        except (OSError, ValueError, TypeError):
            return {}

    def save(self):
        self.data = validate(self.data)
        temporary = self.path + ".tmp"
        with open(temporary, "w") as destination:
            json.dump(self.data, destination)
        try:
            os.rename(temporary, self.path)
        except OSError:
            try:
                os.remove(self.path)
            except OSError:
                pass
            os.rename(temporary, self.path)

    def set_trim(self, matrix):
        self.data["trim_matrix"] = list(matrix)
        self.save()

    def set_controller(self, controller):
        self.data["controller"] = dict(controller)
        self.save()

    def replace(self, candidate):
        """Persist a complete remote configuration after validation."""
        self.data = validate(candidate)
        self.save()

    def reset(self):
        self.data = validate({})
        self.save()

