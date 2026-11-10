import contextlib
import io
import unittest
from unittest.mock import Mock, patch

import go_to
from gps import Pose
from position_controller import PredictiveController


class GoToTests(unittest.TestCase):
    def test_at_target_uses_metres_and_wrapped_degrees(self):
        self.assertEqual(go_to.at_target(Pose(0.,.75,-179.),(0.,.75,179.)),(True,True))
        self.assertEqual(go_to.at_target(Pose(.1,.75,179.),(0.,.75,179.)),(False,True))

    def test_terminal_waypoint_parser(self):
        self.assertEqual(go_to.parse_waypoint('0, .75, 90'),(0.,.75,90.))
        with self.assertRaisesRegex(ValueError,'three values'):
            go_to.parse_waypoint('0 .75')
        with self.assertRaisesRegex(ValueError,'outside'):
            go_to.parse_waypoint('2 0 0')

    def test_saved_car_controller_parameters_are_loaded(self):
        controller=PredictiveController.for_car(1)
        self.assertEqual(controller.kp_dist,4.)
        self.assertEqual(controller.kp_heading,.55)
        self.assertEqual(controller.horizon,.2)

    def test_cli_stops_when_target_is_already_reached(self):
        car=Mock()
        car.__enter__=Mock(return_value=car)
        car.__exit__=Mock(return_value=False)
        car.pose=Pose(0.,.75,90.)
        car.wait_for_position.return_value=car.pose
        car.gps.latest.received_at=1.
        with patch('go_to.Car',return_value=car), patch('go_to.HOLD_TIME',0.), \
                patch('builtins.input',side_effect=['0 .75 90','quit']), \
                contextlib.redirect_stdout(io.StringIO()):
            go_to.main([])
        car.set_velocity.assert_not_called()
        self.assertGreaterEqual(car.stop.call_count,2)
        car.__exit__.assert_called_once()

    def test_invalid_live_input_does_not_move(self):
        car=Mock()
        car.__enter__=Mock(return_value=car);car.__exit__=Mock(return_value=False)
        car.pose=Pose(0.,.75,90.);car.wait_for_position.return_value=car.pose
        with patch('go_to.Car',return_value=car), \
                patch('builtins.input',side_effect=['2 0 0','quit']), \
                contextlib.redirect_stdout(io.StringIO()):
            go_to.main([])
        car.set_velocity.assert_not_called()


if __name__=='__main__':
    unittest.main()
