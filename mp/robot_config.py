"""Validated, persistent per-robot configuration for MicroPython."""
try:
    import ujson as json
except ImportError:
    import json
import math
import os

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
    "trim_matrix": [1., 0., 0., 0., 1., 0., 0., 0., 1.],
    "controller": dict(DEFAULT_CONTROLLER),
}


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def validate(candidate):
    config = dict(DEFAULTS)
    config["controller"] = dict(DEFAULTS["controller"])
    if isinstance(candidate, dict):
        for key in ("version", "car_id", "wifi_ssid", "wifi_password",
                    "telemetry_host", "telemetry_port", "trim_matrix"):
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
    matrix = tuple(config["trim_matrix"])
    if len(matrix) != 9 or not all(_finite(v) and
            (0 if i in (0, 4, 8) else -2) <= v <= 2 for i, v in enumerate(matrix)):
        raise ValueError("invalid trim matrix")
    config["trim_matrix"] = list(matrix)
    controller = config["controller"]
    if not all(_finite(controller.get(key)) and controller[key] > 0
               for key in CONTROLLER_KEYS):
        raise ValueError("invalid controller configuration")
    if controller["max_speed"] > 1 or controller["max_w"] > 1 or not 0 < controller["velocity_alpha"] <= 1:
        raise ValueError("controller limits are out of range")
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

    def reset(self):
        self.data = validate({})
        self.save()

