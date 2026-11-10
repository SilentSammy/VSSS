import unittest

from telemetry_listener import format_packet


class BallFeedbackTests(unittest.TestCase):
    def test_pose_packet_is_readable_when_optional_balls_are_present(self):
        packet = {"board_detected": True, "markers": [{"id": 4, "x": .1, "y": .2, "angle": 90}],
                  "balls": [{"x": .3, "y": .4}]}
        text = format_packet(packet, ('127.0.0.1', 5000), 0)
        self.assertIn('ID 4', text)


if __name__ == '__main__':
    unittest.main()
