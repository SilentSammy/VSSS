"""Control-bag diagnostics align commands, motion samples, and GATT completions."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.ball_control_analysis import analyze


class ControlAnalysisTests(unittest.TestCase):
    def test_near_motion_and_write_duration_are_measured_from_the_bag(self):
        rows = [dict(type='bag_header', provider=dict(vision=dict(metadata=dict(status=dict(
            setup=dict(board=dict(dimensions_m=[1.5, 2.9])))))))]
        for i in range(10):
            now = 10 ** 12 + i * 100_000_000
            common = dict(type='ble', operation='write', characteristic='velocity',
                          car='car1', connection_id='link', request=i)
            payload = dict(value=dict(x=.5, y=0, w=0)) if i < 5 else dict(payload_hex='400000')
            rows.append(dict(common, phase='start', wall_time_ns=now, monotonic_ns=now, **payload))
            rows.append(dict(common, phase='complete', wall_time_ns=now + 125_000_000,
                             monotonic_ns=now + 125_000_000))
            rows.append(dict(type='udp', valid=True, wall_time_ns=now,
                             packet=dict(markers=[dict(id=1, x=0, y=.02 * i)],
                                         balls=[dict(x=0, y=.02 * i + .1)], frame_age_ms=120)))
        rows.append(dict(type='udp', valid=True, wall_time_ns=10 ** 12 + 1_000_000_000,
                         packet=dict(markers=[dict(id=1, x=0, y=.2)], balls=[], frame_age_ms=120)))
        rows.append(dict(type='udp', valid=True, wall_time_ns=10 ** 12 + 1_100_000_000,
                         packet=dict(markers=[dict(id=1, x=0, y=.2)], balls=[dict(x=1.7, y=0)], frame_age_ms=120)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'control.jsonl'
            path.write_text('\n'.join(json.dumps(row) for row in rows))
            report = analyze(path)
        self.assertEqual(report['near_forward_samples'], 10)
        self.assertAlmostEqual(report['near_car_speed_median_mps'], .2)
        self.assertAlmostEqual(report['near_ball_speed_median_mps'], .2)
        self.assertEqual(report['near_forward_command_median'], .5)
        self.assertEqual(report['ble_write_duration_median_ms'], 125)
        self.assertEqual(report['camera_age_median_ms'], 120)
        self.assertEqual(report['no_ball_frames'], 1)
        self.assertEqual(report['raw_outside_ball_detections'], 1)


if __name__ == '__main__':
    unittest.main()
