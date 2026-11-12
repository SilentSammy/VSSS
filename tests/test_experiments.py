"""Experiment safety checks use fake cars and never connect to hardware."""
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ble_protocol import RobotStatus
from tools.run import main


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / 'samples.csv'
        self.car_type = self.enterContext(patch('tools.run.Car'))
        self.gps_type = self.enterContext(patch('tools.run.GPS'))
        self.enterContext(patch('tools.run.time.monotonic', return_value=0))
        self.car = self.car_type.return_value.__enter__.return_value
        self.car.read_trim.return_value = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        self.car.read_controller.return_value = {}
        self.car.read_status.return_value = RobotStatus(3, 3, 0, 0, 0, .5, 90, 10)
        self.car.pose = SimpleNamespace(x=0, y=.5, angle=90)
        self.car.wait_for_position.return_value = self.car.pose
        self.gps_type.return_value.latest = None
        self.args = ['--waypoint', '0', '.5', '90', '--execute',
                     '--output', str(self.output)]

    def test_preview_never_connects(self):
        self.assertEqual(main(['--waypoint', '0', '.5', '90']), 0)
        self.car_type.assert_not_called()
        self.gps_type.assert_not_called()

    def test_completed_waypoint_is_recorded_and_stopped(self):
        self.assertEqual(main(self.args), 0)
        self.car.go_to.assert_called_once_with(0., .5, 90.)
        self.car.stop.assert_called()
        with self.output.open(newline='') as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(rows[0]['mode'], 'reached')
        self.car_type.return_value.__exit__.assert_called_once()

    def test_telemetry_loss_stops_before_disconnect(self):
        self.car.pose = None
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            main(self.args)
        self.car.stop.assert_called_once()
        self.car_type.return_value.__exit__.assert_called_once()
        self.gps_type.return_value.close.assert_called_once()

    def test_starting_outside_bounds_never_sends_waypoint(self):
        self.car.wait_for_position.return_value = SimpleNamespace(x=2, y=0)
        with self.assertRaisesRegex(RuntimeError, 'outside'):
            main(self.args)
        self.car.go_to.assert_not_called()
        self.car.stop.assert_called_once()

    def test_onboard_fault_is_not_reported_as_completion(self):
        self.car.read_status.return_value = RobotStatus(4, 1, 1, 0, 0, 0, 0, 10)
        with self.assertRaisesRegex(RuntimeError, 'waypoint fault: telemetry'):
            main(self.args)
        self.car.stop.assert_called_once()

    def test_existing_recording_is_preserved(self):
        self.output.write_text('previous recording', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            main(self.args)
        self.car.go_to.assert_not_called()
        self.assertEqual(self.output.read_text(encoding='utf-8'), 'previous recording')


if __name__ == '__main__':
    unittest.main()
