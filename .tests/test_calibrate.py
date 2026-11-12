import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vsss._ble_protocol import RobotStatus
from tools.calibrate import calibrate, assess, main, response_between, CalibrationQualityError
from vsss.gps import Pose


class SimulatedRobot:
    number = 1
    is_connected = True

    def __init__(self):
        self.now = 0.
        self.x, self.y, self.heading = 0., 0., 90.
        self.command = (0., 0., 0.)
        self.yaw_gain = 60.
        self.yaw_coast_rate = 0.
        self.coast_left = 0.
        self.trim = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
        self.controller = {'kp_dist': 4.}
        self.listeners = []
        self.seq = 0
        self.writes = []
        self.outage_until = 0.
        self.latest = None
        self.frame()

    @property
    def pose(self):
        if self.now < self.outage_until:
            return None
        return Pose(self.x, self.y, self.heading)

    def frame(self):
        self.seq += 1
        self.latest = SimpleNamespace(received_at=self.now, frame_sequence=self.seq,
                                      frame_age_ms=20., usable=True,
                                      cars={1: self.pose})
        for listener in self.listeners:
            listener(self.latest)

    def sleep(self, seconds):
        import math
        forward = .3*self.command[0] + .03*self.command[1]
        left = .3*self.command[1]
        theta = math.radians(self.heading)
        self.x += seconds * (forward*math.cos(theta)-left*math.sin(theta))
        self.y += seconds * (forward*math.sin(theta)+left*math.cos(theta))
        self.heading += seconds * self.yaw_gain*self.command[2]
        if not self.command[2] and self.coast_left > 0:
            coast_time = min(seconds, self.coast_left)
            self.heading += coast_time * self.yaw_coast_rate
            self.coast_left -= coast_time
        self.now += seconds
        self.frame()

    def add_listener(self, listener): self.listeners.append(listener)
    def remove_listener(self, listener): self.listeners.remove(listener)
    def read_trim(self): return self.trim
    def read_controller(self): return self.controller
    def read_status(self): return RobotStatus(0, 3, 0, 0, self.x, self.y, self.heading, self.seq)
    def wait_for_position(self, timeout): return self.pose
    def set_velocity(self, *command): self.command = command
    def stop(self):
        if self.command[2] and self.yaw_coast_rate:
            self.coast_left = 1.
        self.command = (0., 0., 0.)
    def set_trim(self, matrix): self.writes.append(matrix); return matrix


class CalibrationTests(unittest.TestCase):
    def run_simulation(self, auto_apply, **pulse_settings):
        sim = SimulatedRobot()
        with tempfile.TemporaryDirectory() as temporary, \
                patch('tools.calibrate.time.monotonic', side_effect=lambda: sim.now), \
                patch('tools.calibrate.time.sleep', side_effect=sim.sleep):
            result = calibrate(sim, sim, Path(temporary), auto_apply, **pulse_settings)
            self.assertTrue((Path(temporary) / 'original.json').exists())
            self.assertTrue((Path(temporary) / 'frames.csv').exists())
            self.assertTrue((Path(temporary) / 'proposal.json').exists())
            self.assertEqual((Path(temporary) / 'applied.json').exists(), auto_apply)
        self.assertEqual(sim.command, (0., 0., 0.))
        return result, sim

    def test_simulated_pulses_propose_without_applying(self):
        result, sim = self.run_simulation(False)
        self.assertEqual(result['quality_errors'], [])
        self.assertLess(result['motion_seconds'], 30.)
        self.assertAlmostEqual(result['matrix'][0][1], -.1, places=2)
        self.assertFalse(sim.writes)

    def test_auto_apply_writes_only_after_quality_checks(self):
        result, sim = self.run_simulation(True)
        self.assertEqual(len(sim.writes), 1)
        self.assertEqual(sim.writes[0], result['matrix'])

    def test_stronger_longer_pulses_can_be_configured(self):
        result, _ = self.run_simulation(False, translation_level=.7,
                                        yaw_level=.3, pulse_seconds=1.1)
        self.assertEqual(result['quality_errors'], [])
        self.assertLess(result['motion_seconds'], 30.)

    def test_yaw_duration_can_be_shorter_than_translation(self):
        result, _ = self.run_simulation(False, translation_level=.5,
                                        yaw_level=.2, pulse_seconds=.8, yaw_seconds=.55)
        self.assertEqual(result['quality_errors'], [])

    def test_invalid_pulse_parameters_do_not_connect(self):
        with patch('tools.calibrate.Car') as car_type:
            for arguments in (['--translation-level', '.9'], ['--yaw-level', '.17'],
                              ['--pulse-seconds', '1.3'], ['--yaw-seconds', '0'],
                              ['--center', '0', '1.5', '90']):
                with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                    main(['--car', '1', *arguments])
            car_type.assert_not_called()

    def test_upper_track_center_uses_upper_track_safety_region(self):
        sim = SimulatedRobot()
        sim.y = .75
        sim.frame()
        with tempfile.TemporaryDirectory() as temporary, \
                patch('tools.calibrate.time.monotonic', side_effect=lambda: sim.now), \
                patch('tools.calibrate.time.sleep', side_effect=sim.sleep):
            result = calibrate(sim, sim, Path(temporary), center=(0, .75, 90))
            self.assertEqual(result['quality_errors'], [])
            self.assertGreater(sim.y, .2)
            original = json.loads((Path(temporary) / 'original.json').read_text())
            self.assertEqual(original['center'], [0, .75, 90])

    def test_upper_track_center_rejects_start_below_inner_guard(self):
        sim = SimulatedRobot()
        sim.y = .19
        sim.frame()
        with tempfile.TemporaryDirectory() as temporary, \
                patch('tools.calibrate.time.monotonic', side_effect=lambda: sim.now), \
                patch('tools.calibrate.time.sleep', side_effect=sim.sleep):
            with self.assertRaisesRegex(RuntimeError, 'inner boundary'):
                calibrate(sim, sim, Path(temporary), center=(0, .75, 90))
        self.assertEqual(sim.command, (0., 0., 0.))
        self.assertFalse(sim.writes)

    def test_heading_wrap(self):
        self.assertEqual(response_between(Pose(0, 0, 179), Pose(0, 0, -179))[2], 2.)

    def test_weak_response_rejected(self):
        rows = [dict(axis=i, command=s, frames=4, forward_m=0., left_m=0., heading_deg=0.)
                for i in range(3) for s in (-1, 1)]
        result = {'response_matrix': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                  'matrix': ((1, 0, 0), (0, 1, 0), (0, 0, 1))}
        self.assertTrue(assess(rows, result, result['matrix']))

    def test_disconnect_during_pulse_stops_and_never_applies(self):
        sim = SimulatedRobot()
        def interrupted_sleep(seconds):
            sim.sleep(seconds)
            if sim.now > 1.1:
                sim.is_connected = False
        with tempfile.TemporaryDirectory() as temporary, \
                patch('tools.calibrate.time.monotonic', side_effect=lambda: sim.now), \
                patch('tools.calibrate.time.sleep', side_effect=interrupted_sleep):
            with self.assertRaisesRegex(RuntimeError, 'BLE disconnected'):
                calibrate(sim, sim, Path(temporary), True)
            self.assertTrue((Path(temporary) / 'original.json').exists())
            self.assertFalse((Path(temporary) / 'applied.json').exists())
        self.assertEqual(sim.command, (0., 0., 0.))
        self.assertFalse(sim.writes)

    def test_boundary_violation_stops_and_never_applies(self):
        sim = SimulatedRobot()
        sim.x = .7
        sim.frame()
        with tempfile.TemporaryDirectory() as temporary, \
                patch('tools.calibrate.time.monotonic', side_effect=lambda: sim.now), \
                patch('tools.calibrate.time.sleep', side_effect=sim.sleep):
            with self.assertRaisesRegex(RuntimeError, 'boundary'):
                calibrate(sim, sim, Path(temporary), True)
        self.assertEqual(sim.command, (0., 0., 0.))
        self.assertFalse(sim.writes)

    def test_excessive_pulse_rotation_stops_and_never_applies(self):
        sim = SimulatedRobot()
        sim.yaw_gain = 1000.
        with tempfile.TemporaryDirectory() as temporary, \
                patch('tools.calibrate.time.monotonic', side_effect=lambda: sim.now), \
                patch('tools.calibrate.time.sleep', side_effect=sim.sleep):
            with self.assertRaisesRegex(RuntimeError, 'rotation exceeded'):
                calibrate(sim, sim, Path(temporary), True, yaw_level=.3, pulse_seconds=1.1)
            self.assertFalse((Path(temporary) / 'applied.json').exists())
        self.assertEqual(sim.command, (0., 0., 0.))
        self.assertFalse(sim.writes)

    def test_rotation_during_settling_rejects_run_without_applying(self):
        sim = SimulatedRobot()
        sim.yaw_coast_rate = 150.
        with tempfile.TemporaryDirectory() as temporary, \
                patch('tools.calibrate.time.monotonic', side_effect=lambda: sim.now), \
                patch('tools.calibrate.time.sleep', side_effect=sim.sleep):
            with self.assertRaisesRegex(RuntimeError, 'rotation exceeded.*settling'):
                calibrate(sim, sim, Path(temporary), True)
            self.assertFalse((Path(temporary) / 'applied.json').exists())
        self.assertEqual(sim.command, (0., 0., 0.))
        self.assertFalse(sim.writes)

    def test_short_outage_while_stopped_recovers(self):
        sim = SimulatedRobot()
        triggered = False
        def interrupted_sleep(seconds):
            nonlocal triggered
            sim.sleep(seconds)
            if not triggered and sim.now > 1.5 and not any(sim.command):
                triggered = True
                sim.outage_until = sim.now + .3
        with tempfile.TemporaryDirectory() as temporary, \
                patch('tools.calibrate.time.monotonic', side_effect=lambda: sim.now), \
                patch('tools.calibrate.time.sleep', side_effect=interrupted_sleep):
            result = calibrate(sim, sim, Path(temporary), False)
        self.assertTrue(triggered)
        self.assertEqual(result['quality_errors'], [])
        self.assertFalse(sim.writes)

    def test_outage_during_active_pulse_aborts_and_stops(self):
        sim = SimulatedRobot()
        def interrupted_sleep(seconds):
            sim.sleep(seconds)
            if any(sim.command):
                sim.outage_until = sim.now + 1.
        with tempfile.TemporaryDirectory() as temporary, \
                patch('tools.calibrate.time.monotonic', side_effect=lambda: sim.now), \
                patch('tools.calibrate.time.sleep', side_effect=interrupted_sleep):
            with self.assertRaisesRegex(RuntimeError, 'lost telemetry'):
                calibrate(sim, sim, Path(temporary), True)
        self.assertEqual(sim.command, (0., 0., 0.))
        self.assertFalse(sim.writes)

    def test_incomplete_pulse_is_stopped_discarded_and_retried(self):
        sim = SimulatedRobot()
        triggered = False
        def interrupted_sleep(seconds):
            nonlocal triggered
            sim.sleep(seconds)
            if not triggered and any(sim.command):
                triggered = True
                sim.outage_until = sim.now + .2
        with tempfile.TemporaryDirectory() as temporary, \
                patch('tools.calibrate.time.monotonic', side_effect=lambda: sim.now), \
                patch('tools.calibrate.time.sleep', side_effect=interrupted_sleep):
            result = calibrate(sim, sim, Path(temporary), False)
        self.assertTrue(triggered)
        self.assertEqual(result['samples'], 6)
        self.assertEqual(result['quality_errors'], [])
        self.assertEqual(sim.command, (0., 0., 0.))

    def test_onboard_counter_reset_aborts_without_applying(self):
        sim = SimulatedRobot()
        original_status = sim.read_status
        reads = 0
        def status_with_reset():
            nonlocal reads
            reads += 1
            status = original_status()
            if reads >= 3:
                return RobotStatus(status.mode, status.flags, status.fault, status.reserved,
                                   status.x, status.y, status.angle, 0)
            return status
        sim.read_status = status_with_reset
        with tempfile.TemporaryDirectory() as temporary, \
                patch('tools.calibrate.time.monotonic', side_effect=lambda: sim.now), \
                patch('tools.calibrate.time.sleep', side_effect=sim.sleep):
            with self.assertRaisesRegex(RuntimeError, 'counter reset'):
                calibrate(sim, sim, Path(temporary), True)
            self.assertFalse((Path(temporary) / 'applied.json').exists())
        self.assertEqual(sim.command, (0., 0., 0.))
        self.assertFalse(sim.writes)

    def test_ble_reconnect_invalidates_run_even_if_connected_again(self):
        sim = SimulatedRobot()
        sim.connection_generation = 1
        def reconnecting_sleep(seconds):
            sim.sleep(seconds)
            if sim.now > 1.1:
                sim.connection_generation = 2
        with tempfile.TemporaryDirectory() as temporary, \
                patch('tools.calibrate.time.monotonic', side_effect=lambda: sim.now), \
                patch('tools.calibrate.time.sleep', side_effect=reconnecting_sleep):
            with self.assertRaisesRegex(RuntimeError, 'session reconnected'):
                calibrate(sim, sim, Path(temporary), True)
        self.assertEqual(sim.command, (0., 0., 0.))
        self.assertFalse(sim.writes)

    def test_restore_saved_trim_without_motion(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'original.json'
            matrix = ((1.1, 0, 0), (0, 1, 0), (0, 0, 1))
            path.write_text(json.dumps({'trim': matrix}), encoding='utf-8')
            with patch('tools.calibrate.Car') as car_type, patch('tools.calibrate.GPS'):
                car = car_type.return_value.__enter__.return_value
                car.set_trim.return_value = matrix
                main(['--car', '1', '--restore-trim', str(path)])
                car.stop.assert_called_once()
                car.set_trim.assert_called_once_with(matrix)
                car.set_velocity.assert_not_called()
                car.go_to.assert_not_called()

    def test_restore_rejects_missing_matrix(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'empty.json'
            path.write_text('{}', encoding='utf-8')
            with patch('tools.calibrate.Car') as car_type:
                with self.assertRaises(SystemExit):
                    main(['--car', '1', '--restore-trim', str(path)])
                car_type.assert_not_called()

    def test_apply_saved_proposal_and_reset_do_not_move(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'proposal.json'
            matrix = ((1.1, 0, 0), (0, 1, 0), (0, 0, 1))
            path.write_text(json.dumps({'matrix': matrix, 'quality_errors': []}), encoding='utf-8')
            for args, expected in ((['--apply', str(path)], matrix),
                                   (['--reset'], ((1, 0, 0), (0, 1, 0), (0, 0, 1)))):
                with patch('tools.calibrate.Car') as car_type, patch('tools.calibrate.GPS'):
                    car = car_type.return_value.__enter__.return_value
                    car.set_trim.return_value = expected
                    main(['--car', '1', *args])
                    car.set_trim.assert_called_once_with(expected)
                    car.set_velocity.assert_not_called()
                    car.go_to.assert_not_called()

    def test_apply_rejects_failed_proposal(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'proposal.json'
            path.write_text(json.dumps({'matrix': [[1,0,0],[0,1,0],[0,0,1]],
                                        'quality_errors': ['bad fit']}), encoding='utf-8')
            with patch('tools.calibrate.Car') as car_type:
                with self.assertRaises(SystemExit):
                    main(['--car', '1', '--apply', str(path)])
                car_type.assert_not_called()

    def test_prompt_and_extra_json(self):
        for answer, should_apply in (('y', True), ('n', False)):
            sim = SimulatedRobot()
            with tempfile.TemporaryDirectory() as temporary, \
                    patch('tools.calibrate.time.monotonic', side_effect=lambda: sim.now), \
                    patch('tools.calibrate.time.sleep', side_effect=sim.sleep), \
                    patch('builtins.input', return_value=answer):
                directory = Path(temporary) / 'run'
                directory.mkdir()
                saved = Path(temporary) / 'saved.json'
                result = calibrate(sim, sim, directory, ask_apply=True, save_json=saved)
                self.assertEqual(json.loads(saved.read_text())['matrix'], [list(row) for row in result['matrix']])
                self.assertEqual(bool(sim.writes), should_apply)

    def test_quality_failure_has_clear_message_and_no_traceback(self):
        import io
        from contextlib import redirect_stderr
        with tempfile.TemporaryDirectory() as temporary, \
                patch('tools.calibrate.Car'), patch('tools.calibrate.GPS'), \
                patch('tools.calibrate.calibrate', side_effect=CalibrationQualityError('change 0.164 exceeds 0.15')):
            output = io.StringIO()
            with redirect_stderr(output), self.assertRaises(SystemExit) as error:
                main(['--car', '1', '--output', str(Path(temporary) / 'run')])
            self.assertEqual(error.exception.code, 1)
            self.assertIn('No trim was applied', output.getvalue())
            self.assertIn('change 0.164 exceeds 0.15', output.getvalue())
            self.assertNotIn('Traceback', output.getvalue())


if __name__ == '__main__':
    unittest.main()
