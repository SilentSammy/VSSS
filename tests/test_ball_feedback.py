import unittest

from telemetry_listener import format_packet


class BallFeedbackTests(unittest.TestCase):
    def test_packet_with_marker_and_ball_is_readable(self):
        packet = {"board_detected": True, "markers": [{"id": 4, "x": .1, "y": .2, "angle": 90}],
                  "balls": [{"x": .3, "y": .4}]}
        text = format_packet(packet, ('127.0.0.1', 5000), 0)
        self.assertIn('ID 4', text)
        self.assertIn('ball x=+0.300m y=+0.400m', text)


if __name__ == '__main__':
    unittest.main()
