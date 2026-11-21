"""Kick geometry and bounded full-power motion; no physical provider is used."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import kick_ball as kick
import kick_ball_simple as simple


class KickTests(unittest.TestCase):
    def test_rectangle_route_clears_ball_from_front_or_either_side(self):
        for start in ((0, .5), (.1, .4), (-.1, .4), (.4, -.2)):
            with self.subTest(start=start):
                points = kick.approach(start, (0, 0), (0, 1))
                previous = start
                for point in points:
                    dx, dy = point[0] - previous[0], point[1] - previous[1]
                    t = max(0., min(1., -(previous[0] * dx + previous[1] * dy) /
                                    max(dx * dx + dy * dy, 1e-9)))
                    self.assertGreaterEqual(kick.distance((0, 0), (previous[0] + t * dx, previous[1] + t * dy)), .18)
                    previous = point
                self.assertEqual(points[-1], (0, -.3))

    def test_clear_approach_needs_one_waypoint(self):
        self.assertEqual(kick.approach((0, -.5), (0, 0), (0, 1)), [(0, -.3)])

    def test_full_power_straight_kick_finishes_without_waiting_for_ball(self):
        car = Mock()
        type(car).pose = property(Mock(side_effect=[SimpleNamespace(x=0, y=y) for y in (-.3, -.12, .03)]))
        with patch.object(kick, 'gps', SimpleNamespace(feedback_problem=None, board_detected=True)), patch.object(kick, 'sleep'):
            kick.strike(car, (0, 0), (0, 1), (1.5, 2.9))
        self.assertEqual(car.set_velocity.call_count, 2)
        self.assertTrue(all(call.kwargs == dict(x=1., y=0., w=0.) for call in car.set_velocity.call_args_list))
        car.stop.assert_called_once()

    def test_kick_stops_on_feedback_loss_or_timeout(self):
        for pose in (None, SimpleNamespace(x=0, y=-.3)):
            car = Mock(pose=pose)
            clock = iter([0., 0., 2.])
            with self.subTest(pose=pose), patch.object(kick, 'gps', SimpleNamespace(feedback_problem=None, board_detected=True)), \
                    patch.object(kick, 'sleep'), patch.object(kick, 'monotonic', side_effect=lambda: next(clock)), \
                    self.assertRaises((RuntimeError, TimeoutError)):
                kick.strike(car, (0, 0), (0, 1), (1.5, 2.9))
            car.stop.assert_called_once()

    def test_route_outside_field_is_rejected_before_moving(self):
        car = Mock()
        with self.assertRaisesRegex(RuntimeError, 'field space'):
            kick.line_up(car, [(.8, 0)], (0, 0), 90, (1.5, 2.9))
        car.go_to.assert_not_called()

    def test_moving_ball_aborts_before_strike(self):
        car = Mock()
        with patch.object(kick, 'go_to'), patch.object(kick, 'observe', return_value=(SimpleNamespace(angle=90), (.1, 0))):
            with self.assertRaisesRegex(RuntimeError, 'ball moved'):
                kick.line_up(car, [(0, -.3)], (0, 0), 90, (1.5, 2.9))

    def test_happy_path_only_lines_up_once_and_strikes_once(self):
        car = Mock()
        with patch.object(simple, 'aim', return_value=(None, (0, 0), (0, 1), 90)), \
                patch.object(simple, 'line_up') as line_up, patch.object(simple, 'strike') as strike:
            simple.kick(car, (0, 1))
        line_up.assert_called_once_with(car, [(0., -.3)], (0, 0), 90, (1.5, 2.9))
        strike.assert_called_once()
        car.stop.assert_called_once()


if __name__ == '__main__':
    unittest.main()
