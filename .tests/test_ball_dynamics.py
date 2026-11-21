import json
import math
from pathlib import Path
import tempfile
import unittest

from tools.ball_dynamics import BallModel, analyze_slow_rolls, detect_rolls, load_samples


class BallDynamicsTests(unittest.TestCase):
    def test_slow_roll_report_distinguishes_quiet_detector_steps(self):
        rows = [dict(type='udp', valid=True, wall_time_ns=10 ** 12 + index * 100_000_000,
                     packet=dict(markers=[dict(id=1, x=0, y=0)], balls=[dict(x=.4, y=0)]))
                for index in range(20)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'quiet.jsonl'
            path.write_text('\n'.join(json.dumps(row) for row in rows))
            report = analyze_slow_rolls(path)
        self.assertEqual(report['stationary_frame_steps'], 19)
        self.assertEqual(report['stationary_step_median_m'], 0)
        self.assertEqual(report['stationary_step_p95_m'], 0)
        self.assertEqual(report['terminal_episodes'], [])

    def test_impact_transfers_only_closing_normal_velocity(self):
        model = BallModel(.8, .1)
        self.assertEqual(model.impact_velocity((-.4, 0), (0, 0), (1, 0)), (0, 0))
        vx, vy = model.impact_velocity((.5, .2), (.1, .2), (1, 0))
        self.assertAlmostEqual(vx, .42)
        self.assertAlmostEqual(vy, .2)

    def test_constant_resistance_predicts_expected_stop(self):
        model = BallModel(1, .5, stop_speed=.05)
        x, y = model.stopping_point((0, 0), (1, 0))
        self.assertAlmostEqual(x, (1 - .05 ** 2) / 1.0, delta=.015)
        self.assertAlmostEqual(y, 0, places=6)

    def test_loader_rejects_outlier_and_detects_release(self):
        start = 1_000_000_000_000
        rows = []
        # Contact for three frames, then the ball rolls away and slows.
        points = [(0.10, 0), (0.11, 0), (0.12, 0),
                  (0.23, 0), (0.31, 0), (0.37, 0), (0.40, 0), (0.41, 0)]
        for index, (ball_x, ball_y) in enumerate(points):
            rows.append({'type': 'udp', 'valid': True,
                         'wall_time_ns': start + index * 100_000_000,
                         'packet': {'markers': [{'id': 1, 'x': 0, 'y': 0}],
                                    'balls': [{'x': ball_x, 'y': ball_y}]}})
        # A detector jump outside the plausible field is ignored.
        rows.insert(5, {'type': 'udp', 'valid': True,
                        'wall_time_ns': start + 450_000_000,
                        'packet': {'markers': [{'id': 1, 'x': 0, 'y': 0}],
                                   'balls': [{'x': 9, 'y': 9}]}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bag.jsonl'
            path.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
            samples = load_samples(path)
        self.assertEqual(len(samples), len(points))
        rolls = detect_rolls(samples, minimum_distance=.05, minimum_duration=.2)
        self.assertTrue(rolls)
        self.assertGreater(rolls[0].distance, .1)


if __name__ == '__main__':
    unittest.main()
