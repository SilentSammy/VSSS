import json
from pathlib import Path
import tempfile
import unittest

from mp.robot_config import RobotConfig, validate


class RobotConfigTests(unittest.TestCase):
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

    def test_rejects_unsafe_limits(self):
        candidate = json.loads(json.dumps(validate({})))
        candidate['controller']['max_speed'] = 1.1
        with self.assertRaises(ValueError):
            validate(candidate)


if __name__ == '__main__':
    unittest.main()
