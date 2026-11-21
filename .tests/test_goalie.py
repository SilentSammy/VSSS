"""The happy-path goalie predicts one straight crossing and commands X only."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import goalie


class GoalieTests(unittest.TestCase):
    def test_projects_a_diagonal_shot_to_the_goalie_line(self):
        self.assertAlmostEqual(goalie.intercept((0, -.2, -.5), (.1, -.18, -.42), 1.2), .225)

    def test_ignores_repeated_receding_or_already_passed_shots_and_clamps_target(self):
        self.assertIsNone(goalie.intercept((0, 0, 0), (0, .1, .1), 1.2))
        self.assertIsNone(goalie.intercept((0, 0, 0), (.1, .1, -.1), 1.2))
        self.assertIsNone(goalie.intercept((0, 0, 1.1), (.1, .1, 1.3), 1.2))
        self.assertEqual(goalie.intercept((0, 0, 0), (.1, .5, .1), 1.2), .63)

    def test_defender_never_commands_y_or_rotation_and_always_stops(self):
        pose = SimpleNamespace(x=0, y=1.2, angle=0)
        car = Mock(pose=pose)
        car.wait_for_position.return_value = pose
        frames = [SimpleNamespace(board_detected=True, balls=[SimpleNamespace(x=x,y=y)],
                    server_timestamp=t) for t,x,y in ((0, -.2,-.5), (.1,-.18,-.42), (.2,-.16,-.34))]
        now = [0.]
        gps = SimpleNamespace(feedback_problem=None, latest=None)
        def sleep(dt):
            now[0] += dt
            gps.latest = next(observations, frames[-1])
        observations = iter(frames)
        gps.latest = next(observations)
        with patch.object(goalie, 'gps', gps), patch.object(goalie, 'sleep', side_effect=sleep), \
                patch.object(goalie, 'monotonic', side_effect=lambda:now[0]):
            goalie.defend(car, duration=.12)
        commands = [c.kwargs for c in car.set_velocity.call_args_list]
        self.assertTrue(all(c['y'] == c['w'] == 0 for c in commands))
        self.assertTrue(any(c['x'] > 0 for c in commands))
        car.stop.assert_called_once()


if __name__ == '__main__':
    unittest.main()
