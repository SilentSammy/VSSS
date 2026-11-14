import math
import unittest

from mp.onboard_controller import OnboardController


class OnboardControllerTests(unittest.TestCase):
    def test_stops_without_target_or_pose(self):
        controller = OnboardController()
        self.assertEqual(controller.update((0, 0, 0), 0), (0, 0, 0))
        controller.set_target(1, 0, 0)
        self.assertEqual(controller.update(None, 10), (0, 0, 0))

    def test_field_command_is_rotated_into_robot_frame(self):
        controller = OnboardController(max_speed=.4)
        controller.set_target(1, 0, math.pi / 2)
        x, y, w = controller.update((0, 0, 90), 0)
        self.assertAlmostEqual(x, 0, places=6)
        self.assertAlmostEqual(y, -.4, places=6)
        self.assertAlmostEqual(w, 0, places=6)

    def test_prediction_brakes_a_moving_pose(self):
        controller = OnboardController(kp_dist=1, max_speed=1, horizon=.2,
                                       velocity_alpha=1)
        controller.set_target(1, 0, 0)
        controller.update((0, 0, 0), 0)
        x, _, _ = controller.update((.1, 0, 0), 100)
        self.assertAlmostEqual(x, .7)

    def test_target_requires_continuous_hold_before_reached(self):
        controller = OnboardController(hold_ms=600)
        controller.set_target(0, 0, 0)
        self.assertEqual(controller.update((0, 0, 0), 100), (0, 0, 0))
        self.assertFalse(controller.reached(699))
        self.assertTrue(controller.reached(700))

    def test_near_target_does_not_apply_discontinuous_minimum_kick(self):
        controller = OnboardController(kp_dist=.1, min_speed=.12,
                                       position_tolerance=.025, horizon=0)
        controller.set_target(.03, 0, 0)
        x, y, _ = controller.update((0, 0, 0), 0)
        self.assertAlmostEqual(x, .003)
        self.assertEqual(y, 0)

        controller = OnboardController(kp_heading=.1, min_w=.08,
                                       heading_tolerance_deg=5, horizon=0)
        controller.set_target(0, 0, math.radians(6))
        _, _, w = controller.update((0, 0, 0), 0)
        self.assertAlmostEqual(w, math.radians(.6))

    def test_logged_fast_pass_through_arrival_band_keeps_braking(self):
        controller = OnboardController()
        controller.set_target(0, .5, math.pi / 2)
        controller.velocity = [.0217, .2415, .5471]
        x, _, w = controller.update((.01, .5, 91), 4709)
        self.assertLess(x, 0)
        self.assertLess(w, 0)
        self.assertIsNone(controller.stable_since)
        self.assertFalse(controller.reached(5400))

    def test_arrival_hold_starts_only_after_estimated_speed_settles(self):
        controller = OnboardController()
        controller.set_target(0, 0, 0)
        controller.previous = (0, -.04, 0, 0)
        x, _, _ = controller.update((0, 0, 0), 100)
        self.assertLess(x, 0)
        self.assertIsNone(controller.stable_since)
        for now in range(200, 1001, 100):
            controller.update((0, 0, 0), now)
            if now < 1000:
                self.assertFalse(controller.reached(now))
        self.assertTrue(controller.reached(1000))

    def test_missing_pose_cancels_arrival_hold(self):
        controller = OnboardController()
        controller.set_target(0, 0, 0)
        controller.update((0, 0, 0), 100)
        controller.update(None, 200)
        self.assertFalse(controller.reached(1000))


if __name__ == "__main__":
    unittest.main()
