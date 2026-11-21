"""The student example uses only public APIs and stops when observation fails."""
import ast
from contextlib import redirect_stderr
import io
from itertools import chain, repeat
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import ball_to_zone as example


class BallToZoneTests(unittest.TestCase):
    def test_forward_feed_does_not_collapse_at_bumper_contact(self):
        pose = SimpleNamespace(x=0, y=-.09, angle=90)
        command = example.push_velocity(pose, (0, 0), (0, 1))
        self.assertAlmostEqual(command['x'], .4)
        self.assertAlmostEqual(command['y'], 0)

    def test_moderate_heading_error_keeps_forward_motion_and_bounds_corrections(self):
        pose = SimpleNamespace(x=0, y=-.09, angle=70)
        command = example.push_velocity(pose, (0, 0), (0, 1))
        self.assertGreater(command['x'], .15)
        self.assertLessEqual(command['x'], .4)
        self.assertLessEqual(abs(command['y']), .1)
        self.assertLessEqual(abs(command['w']), .15)

    def test_speed_tapers_near_goal_without_entering_old_creep_regime(self):
        near = example.push_velocity(SimpleNamespace(x=0, y=.71, angle=90), (0, .8), (0, 1))
        far = example.push_velocity(SimpleNamespace(x=0, y=-.09, angle=90), (0, 0), (0, 1))
        self.assertGreaterEqual(near['x'], .3)
        self.assertLess(near['x'], far['x'])

    def test_offset_ball_keeps_moderate_forward_capture_without_centering_switch(self):
        pose = SimpleNamespace(x=-.03, y=-.23, angle=90)
        command = example.push_velocity(pose, (0, 0), (0, 1))
        self.assertGreater(command['x'], .25)
        self.assertLess(command['y'], 0)
        self.assertLessEqual(abs(command['y']), .15)
        capture = example.push_velocity(SimpleNamespace(x=0, y=-.23, angle=90), (0, 0), (0, 1))
        self.assertEqual(capture['x'], .3)

    def test_recorded_bag8_first_push_uses_moderate_capture(self):
        # Latest visible pose at bag 8's first x=0.5 command, 21.89 s.
        pose = SimpleNamespace(x=.142, y=-1.094, angle=98)
        command = example.push_velocity(pose, (.143, -.86), (0, 1), field=(1.5, 2.9))
        self.assertGreater(command['x'], .25)
        self.assertLessEqual(command['x'], .3)
        self.assertLess(command['y'], 0)

    def test_bag12_centering_threshold_crossings_do_not_interrupt_forward_feed(self):
        # Attempt 6, 307.65..309.97 s: the old controller alternated zero/forward.
        frames = [(-.05, .935, -82, -.019, .686), (-.051, .936, -83, -.019, .685),
                  (-.034, .866, -96, -.019, .685), (-.035, .8, -100, -.022, .67),
                  (-.024, .801, -95, -.024, .647), (-.021, .79, -90, -.026, .634)]
        for x, y, angle, bx, by in frames:
            with self.subTest(angle=angle):
                command = example.push_velocity(SimpleNamespace(x=x, y=y, angle=angle), (bx, by), (0, 0))
                self.assertIsNotNone(command)
                self.assertGreater(command['x'], .25)
                self.assertLessEqual(abs(command['w']), .05)

    def test_onboard_stale_pose_is_reported_separately_from_a_controller_fault(self):
        car = Mock()
        car.read_status.return_value = SimpleNamespace(fault=0, pose_fresh=False, fault_name='none')
        with patch.object(example, 'observe', return_value=(SimpleNamespace(x=0, y=0), (0, .3))):
            with self.assertRaisesRegex(RuntimeError, 'Onboard camera pose is stale'):
                example.go_to(car, (0, -.23), 90, math.inf)

    def test_rotation_is_locked_at_contact_and_remains_locked_as_ball_separates(self):
        pose = SimpleNamespace(x=0, y=-.1, angle=80)
        self.assertEqual(example.push_velocity(pose, (0, 0), (0, 1))['w'], 0)
        # A rolling ball may open a gap again; this must not re-enable a turn.
        pose = SimpleNamespace(x=0, y=-.23, angle=80)
        self.assertGreater(example.push_velocity(pose, (0, 0), (0, 1))['w'], 0)
        self.assertEqual(example.push_velocity(pose, (0, 0), (0, 1), contact=True)['w'], 0)

    def test_glancing_contact_and_outward_boundary_push_are_rejected(self):
        self.assertIsNone(example.push_velocity(SimpleNamespace(x=.07, y=-.09, angle=90), (0, 0), (0, 1)))
        self.assertIsNone(example.push_velocity(SimpleNamespace(x=.63, y=-.1, angle=80),
                                               (.65, 0), (.6, 1), field=(1.5, 2.9)))

    def test_outside_ball_stops_before_planning_a_chase(self):
        car = Mock()
        with patch.object(example, 'observe', return_value=(SimpleNamespace(x=0, y=0), (-.76, .6))):
            with self.assertRaisesRegex(RuntimeError, 'Ball is at or outside'):
                example.move_ball(car, field=(1.5, 2.9))
        car.go_to.assert_not_called()
        car.set_velocity.assert_not_called()
        car.stop.assert_called_once()

    def test_real_loss_of_alignment_requires_repositioning(self):
        for pose in (SimpleNamespace(x=0, y=.1, angle=90),
                     SimpleNamespace(x=.2, y=-.09, angle=90),
                     SimpleNamespace(x=0, y=-.09, angle=20)):
            with self.subTest(pose=pose):
                self.assertIsNone(example.push_velocity(pose, (0, 0), (0, 1)))

    def test_replanning_waits_for_a_coasting_ball_to_settle(self):
        clock = [0.]
        def sleep(seconds):
            clock[0] += seconds
        def observe(car):
            return SimpleNamespace(x=0, y=-.3), (min(.1, .2 * clock[0]), 0)
        with patch.object(example, 'observe', side_effect=observe), \
                patch.object(example, 'sleep', side_effect=sleep), \
                patch.object(example, 'monotonic', side_effect=lambda: clock[0]):
            example.wait_for_still_ball(Mock(), 5)
        self.assertGreaterEqual(clock[0], 1.1)

    def test_firm_push_releases_at_zone_entrance_to_allow_for_coast(self):
        clock = [0.]
        car = Mock()
        far = (SimpleNamespace(x=0, y=-.3, angle=90), (0, 0))
        pushing = (SimpleNamespace(x=0, y=.01, angle=90), (0, .1))
        separated = (SimpleNamespace(x=0, y=.5, angle=80), (0, .75))
        entering = (SimpleNamespace(x=0, y=.75, angle=90), (0, .84))
        observations = iter(chain([far, pushing, separated], repeat(entering)))
        def sleep(seconds):
            clock[0] += seconds
        with patch.object(example, 'observe', side_effect=lambda car: next(observations)), \
                patch.object(example, 'go_to'), \
                patch.object(example, 'push_velocity', wraps=example.push_velocity) as push, \
                patch.object(example, 'sleep', side_effect=sleep), \
                patch.object(example, 'monotonic', side_effect=lambda: clock[0]):
            result = example.move_ball(car)
        self.assertEqual(result, (0, .84))
        self.assertEqual(car.set_velocity.call_count, 2)
        self.assertTrue(all(call.kwargs['contact'] for call in push.call_args_list))
        self.assertTrue(all(call.kwargs['w'] == 0 for call in car.set_velocity.call_args_list))

    def test_example_imports_only_student_api_and_standard_library(self):
        tree = ast.parse(Path(example.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module.startswith('vsss'):
                self.assertEqual(node.module, 'vsss')
                self.assertTrue({name.name for name in node.names} <= {'Car', 'gps', 'run'})
            if isinstance(node, ast.Import):
                self.assertFalse(any(name.name.startswith('vsss.') for name in node.names))

    def test_approach_goes_around_ball_instead_of_crossing_it(self):
        start = (.1, .5)
        points = example.approach_points(start, (0, 0), (0, 1))
        self.assertGreater(len(points), 1)
        previous = start
        for point in points:
            dx, dy = point[0] - previous[0], point[1] - previous[1]
            denominator = dx * dx + dy * dy
            t = max(0, min(1, -(previous[0] * dx + previous[1] * dy) / max(denominator, 1e-9)))
            self.assertGreaterEqual(math.hypot(previous[0] + t * dx, previous[1] + t * dy), .16)
            previous = point
        self.assertAlmostEqual(points[-1][0], 0)
        self.assertAlmostEqual(points[-1][1], -.23)

    def test_straight_approach_needs_only_one_waypoint(self):
        self.assertEqual(example.approach_points((0, -.5), (0, 0), (0, 1)), [(0, -.23)])

    def test_observation_failure_stops_motion(self):
        car = Mock()
        with patch.object(example, 'observe', side_effect=RuntimeError('feedback lost')):
            with self.assertRaisesRegex(RuntimeError, 'feedback lost'):
                example.move_ball(car)
        car.stop.assert_called_once()

    def test_no_room_behind_ball_is_reported_before_driving(self):
        car = Mock()
        with patch.object(example, 'observe', return_value=(SimpleNamespace(x=0, y=-1), (0, -1.35))):
            with self.assertRaisesRegex(RuntimeError, 'Not enough room'):
                example.move_ball(car)
        car.go_to.assert_not_called()
        car.set_velocity.assert_not_called()
        car.stop.assert_called_once()

    def test_ball_already_in_zone_is_checked_for_settling_without_driving(self):
        car = Mock()
        now = [0.]
        def sleep(seconds):
            now[0] += seconds
        with patch.object(example, 'observe', return_value=(SimpleNamespace(x=0, y=0), (.12, 1))), \
                patch.object(example, 'monotonic', side_effect=lambda: now[0]), \
                patch.object(example, 'sleep', side_effect=sleep):
            self.assertEqual(example.move_ball(car), (.12, 1))
        self.assertGreaterEqual(now[0], 1.2)
        car.go_to.assert_not_called()
        car.set_velocity.assert_not_called()

    def test_invalid_goal_is_rejected_without_connecting_a_car(self):
        for arguments in (['--target', '0', '5'], ['--push-speed', '1'], ['--push-speed', 'nan']):
            with self.subTest(arguments=arguments), \
                    patch.object(example, 'Car') as constructor, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    example.main(arguments)
            constructor.assert_not_called()


if __name__ == '__main__':
    unittest.main()
