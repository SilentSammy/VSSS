"""Flash recorder bounds, recovery, sampling, and storage-failure isolation."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mp.navigation_log import COLUMNS, SAMPLE_SIZE, NavigationLog


class NavigationLogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.prefix = str(Path(self.temp.name) / 'navigation')
        self.settings = dict(car_id=1, controller={'horizon': .2}, trim_matrix=[1,0,0,0,1,0,0,0,1])

    def read(self, path):
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line]

    def test_rotation_is_bounded_and_every_segment_has_schema(self):
        log = NavigationLog(self.settings, prefix=self.prefix, max_bytes=2048, buffer_bytes=256)
        for index in range(120):
            log.event('test', index=index, text='x' * 50)
            log.flush()
        log.close()
        files = list(Path(self.temp.name).glob('navigation*.jsonl'))
        self.assertEqual(len(files), 3)
        for path in files:
            self.assertLessEqual(path.stat().st_size, 2048)
            self.assertEqual(self.read(path)[0]['type'], 'schema')
        events = [row for path in files for row in self.read(path) if row['type'] == 'test']
        self.assertEqual(max(row['index'] for row in events), 119)

    def test_boot_appends_without_erasing_previous_runs_and_fences_torn_tail(self):
        first = NavigationLog(self.settings, prefix=self.prefix)
        first.event('waypoint', target=[0, .4, 90])
        first.close()
        with open(self.prefix + '.jsonl', 'a') as file:
            file.write('{"torn":')
        second = NavigationLog(self.settings, prefix=self.prefix)
        second.close()
        rows = []
        for line in Path(self.prefix + '.jsonl').read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
        self.assertEqual(sum(row['type'] == 'boot' for row in rows), 2)
        self.assertTrue(any(row['type'] == 'waypoint' for row in rows))
        self.assertEqual(len({row['boot_id'] for row in rows if row['type'] == 'schema'}), 2)

    def sample_inputs(self):
        motor = SimpleNamespace(pwm1=None, pwm2=None)
        car = SimpleNamespace(x=0, y=0, w=0, motor_fl=motor, motor_fr=motor, motor_bl=motor, motor_br=motor)
        controller = SimpleNamespace(velocity=[.1, .2, .3])
        receiver = SimpleNamespace(pose=(0, .3, 90), received_ms=900, frame_age_ms=120,
                                   frame_sequence=7, packets=8, parse_errors=0, fresh=True)
        state = dict(mode=2, fault=0, waypoint_id=1, waypoint=(0, .4, 90), control_dt_ms=100,
                     control_raw=(.2, 0, 0), control_command=(.21, 0, 0))
        return state, receiver, controller, car

    def test_navigation_is_decimated_idle_is_quiet_and_fault_flushes(self):
        log = NavigationLog(self.settings, prefix=self.prefix)
        state, receiver, controller, car = self.sample_inputs()
        for now in range(1000, 2000, 10):
            log.sample(now, state, receiver, controller, car, True, 10)
        state.update(mode=4, fault=2)
        log.sample(2000, state, receiver, controller, car, True, 10)
        rows = self.read(log.path(0))
        samples = [row for row in rows if row['type'] == 'sample']
        self.assertEqual(len(samples), 6)
        self.assertEqual(len(samples[-1]['v']), len(COLUMNS))
        self.assertEqual(dict(zip(COLUMNS, samples[-1]['v']))['fault'], 2)
        for now in range(2010, 3010, 10):
            log.sample(now, state, receiver, controller, car, True, 10)
        log.close()
        self.assertEqual(len([row for row in self.read(log.path(0)) if row['type'] == 'sample']), 6)

    def test_io_failure_disables_logging_without_raising(self):
        log = NavigationLog(self.settings, prefix=self.prefix)
        log.event('test')
        with patch.object(log, '_write', side_effect=OSError('flash full')):
            log.flush()
        self.assertFalse(log.enabled)
        log.sample(1000, *self.sample_inputs(), True, 10)
        log.event('still safe')
        log.close()

    def test_active_ring_never_writes_flash_and_keeps_latest_samples(self):
        log = NavigationLog(self.settings, prefix=self.prefix, sample_slots=8)
        state, receiver, controller, car = self.sample_inputs()
        with patch.object(log, '_write', side_effect=AssertionError('flash during motion')) as write:
            for now in range(1000, 5000, 200):
                log.sample(now, state, receiver, controller, car, True, 10)
            write.assert_not_called()
            self.assertTrue(log.enabled)
            self.assertEqual(log.count, 8)
        state.update(mode=4, fault=2)
        log.sample(5000, state, receiver, controller, car, True, 10)
        rows = self.read(log.path(0))
        samples = [row for row in rows if row['type'] == 'sample']
        self.assertEqual(len(samples), 8)
        self.assertEqual(samples[0]['v'][0], 3600)
        self.assertEqual(samples[-1]['v'][0], 5000)
        self.assertEqual(rows[-1]['samples_dropped'], 13)
        log.close()

    def test_idle_without_telemetry_records_null_fields(self):
        log = NavigationLog(self.settings, prefix=self.prefix)
        state, _, controller, car = self.sample_inputs()
        state.update(mode=0, waypoint=None, control_dt_ms=None)
        log.sample(1000, state, None, controller, car, False, 10)
        row = next(row for row in self.read(log.path(0)) if row['type'] == 'sample')
        values = dict(zip(COLUMNS, row['v']))
        self.assertIsNone(values['pose_x'])
        self.assertIsNone(values['target_x'])
        self.assertIsNone(values['control_dt_ms'])
        self.assertEqual(values['flags'], 0)
        log.close()

    def test_heap_budget_keeps_reserve(self):
        available = 20 * 1024 + 64 * SAMPLE_SIZE
        with patch('mp.navigation_log.gc.mem_free', return_value=available, create=True):
            log = NavigationLog(self.settings, prefix=self.prefix)
        self.assertEqual(log.sample_slots, 64)
        self.assertEqual(available - len(log.ring), 20 * 1024)
        log.close()

    def test_switch_to_moving_manual_control_does_not_write_flash(self):
        log = NavigationLog(self.settings, prefix=self.prefix)
        state, receiver, controller, car = self.sample_inputs()
        with patch.object(log, '_write', side_effect=AssertionError('flash during motion')) as write:
            log.sample(1000, state, receiver, controller, car, True, 10)
            state['mode'], car.x = 1, .2
            log.sample(1200, state, receiver, controller, car, True, 10)
            write.assert_not_called()
            self.assertTrue(log.enabled)
        state['mode'], car.x = 0, 0
        log.sample(1400, state, receiver, controller, car, True, 10)
        self.assertEqual(log.count, 0)
        log.close()


if __name__ == '__main__':
    unittest.main()
