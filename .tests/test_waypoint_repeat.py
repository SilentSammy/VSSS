"""Oscillation classification independent of physical hardware."""
import unittest
import contextlib
import io
import json
from unittest.mock import patch
from tools.waypoint_repeat import main, oscillating, RecoveryWindow


class OscillationTests(unittest.TestCase):
    def test_recovery_window_tolerates_one_frame_and_expires(self):
        window = RecoveryWindow(grace=2.)
        self.assertEqual(window.update(True, 0), ('fresh', 0.))
        self.assertEqual(window.update(False, 1), ('missing', 0.))
        state, elapsed = window.update(True, 1.1)
        self.assertEqual(state, 'recovered')
        self.assertAlmostEqual(elapsed, .1)
        self.assertEqual(window.update(False, 2), ('missing', 0.))
        state, elapsed = window.update(False, 4.1)
        self.assertEqual(state, 'expired')
        self.assertAlmostEqual(elapsed, 2.1)

    def test_repeated_crossings_are_flagged(self):
        samples = [(i * .1, 0., .4 + (.05 if (i // 7) % 2 else -.05), 90.) for i in range(60)]
        self.assertTrue(oscillating(samples, (0., .4, 90.)))

    def test_camera_noise_and_a_normal_approach_are_not_flagged(self):
        noise = [(i * .1, 0., .4 + (.01 if i % 2 else -.01), 90.) for i in range(60)]
        approach = [(i * .1, 0., -.4 + i * .8 / 59, 90.) for i in range(60)]
        self.assertFalse(oscillating(noise, (0., .4, 90.)))
        self.assertFalse(oscillating(approach, (0., .4, 90.)))

    def test_heading_wraparound_is_not_an_oscillation(self):
        samples = [(i * .1, 0., .4, -179. if i % 2 else 179.) for i in range(60)]
        self.assertFalse(oscillating(samples, (0., .4, 180.)))

    def test_custom_circuit_preview_does_not_connect(self):
        with patch('tools.waypoint_repeat.Car') as car, patch('tools.waypoint_repeat.GPS') as gps, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            main(['--target', '0', '.5', '90', '--batches', '1'])
        self.assertEqual(json.loads(output.getvalue())['targets'], [[0, .5, 90]])
        car.assert_not_called()
        gps.assert_not_called()

    def test_custom_target_outside_guard_is_rejected_before_connection(self):
        with patch('tools.waypoint_repeat.Car') as car, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(['--execute', '--target', '.6', '0', '90'])
        car.assert_not_called()


if __name__ == '__main__':
    unittest.main()
