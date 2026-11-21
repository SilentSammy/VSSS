import json
from pathlib import Path
import tempfile
import unittest

from mp.robot_config import RobotConfig, validate
from mp.onboard_controller import CONTROLLER_KEYS as FIRMWARE_CONTROLLER_KEYS
from vsss._ble_protocol import CONTROLLER_KEYS as HOST_CONTROLLER_KEYS
from vsss._ble_protocol import normalize_robot_config


class RobotConfigTests(unittest.TestCase):
    def test_host_and_firmware_controller_layouts_match(self):
        self.assertEqual(HOST_CONTROLLER_KEYS, FIRMWARE_CONTROLLER_KEYS)

    def test_invalid_file_uses_safe_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'robot.json'
            path.write_text('{broken', encoding='utf-8')
            config = RobotConfig(str(path))
            self.assertEqual(config.data['car_id'], 1)
            self.assertEqual(config.data['trim_matrix'][0], 1.)
            self.assertEqual(config.data['telemetry_host'], '192.168.12.1')
            self.assertEqual(config.data['telemetry_port'], 5000)

    def test_trim_and_controller_survive_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'robot.json'
            config = RobotConfig(str(path))
            config.set_trim((1.1, 0, 0, 0, 1, 0, 0, 0, .9))
            controller = dict(config.data['controller'])
            controller['kp_dist'] = 3.5
            config.set_controller(controller)
            loaded = RobotConfig(str(path))
            self.assertEqual(loaded.data['trim_matrix'][0], 1.1)
            self.assertEqual(loaded.data['controller']['kp_dist'], 3.5)
            self.assertFalse(Path(str(path) + '.tmp').exists())

    def test_old_controller_file_gains_new_defaults(self):
        current = validate({})
        legacy = dict(current)
        legacy['controller'] = {key: value for key, value in current['controller'].items()
                                if key not in ('position_hysteresis', 'heading_hysteresis',
                                               'telemetry_recovery_ms', 'waypoint_timeout_ms')}
        migrated = validate(legacy)
        self.assertEqual(migrated['controller']['position_hysteresis'], 1.5)
        self.assertEqual(migrated['controller']['telemetry_recovery_ms'], 3000)

    def test_rejects_unsafe_limits(self):
        candidate = json.loads(json.dumps(validate({})))
        candidate['controller']['max_speed'] = 1.1
        with self.assertRaises(ValueError):
            validate(candidate)
        for key, value in (('position_hysteresis', .9),
                           ('telemetry_recovery_ms', 99),
                           ('waypoint_timeout_ms', 999)):
            candidate = json.loads(json.dumps(validate({})))
            candidate['controller'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(candidate)

    def test_motor_pins_are_persisted_and_must_be_unique_esp32c3_gpio(self):
        candidate = validate({})
        candidate['motor_pins'] = [[2, 3], [4, 5], [6, 7], [8, 9]]
        self.assertEqual(validate(candidate)['motor_pins'][0], [2, 3])
        candidate['motor_pins'][3][1] = 2
        with self.assertRaises(ValueError):
            validate(candidate)

    def test_host_rejects_incomplete_or_invalid_remote_configuration(self):
        candidate = validate({})
        self.assertEqual(normalize_robot_config(candidate)['motor_pins'][3], [8, 9])
        del candidate['motor_pins']
        with self.assertRaises(ValueError):
            normalize_robot_config(candidate)


if __name__ == '__main__':
    unittest.main()
