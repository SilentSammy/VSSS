from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from detection.ball_det import BallDetector
from detection.main import argument_parser, localize_balls
from telemetry_listener import format_packet


class BallFeedbackTests(unittest.TestCase):
    def test_ball_detection_can_be_disabled_for_headless_throughput(self):
        args=argument_parser().parse_args(['--headless','--no-ball-detection'])
        self.assertTrue(args.headless)
        self.assertTrue(args.no_ball_detection)

    def test_blue_ball_detection_and_projection(self):
        hsv = np.uint8([[[107, 120, 180]]])
        colour = tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
        frame = np.full((200, 300, 3), 240, np.uint8)
        cv2.circle(frame, (160, 100), 25, colour, -1)
        detections = BallDetector().detect(frame)
        self.assertEqual(len(detections), 1)
        self.assertLess(np.linalg.norm(np.array(detections[0].centroid) - (160, 100)), 2)
        calls = []

        def project(pnp, centroid, shape, z):
            calls.append(z)
            return .25, -.5

        estimator = SimpleNamespace(project_point_to_board=project)
        balls = localize_balls(detections, estimator, (None, object()), frame, .02)
        self.assertEqual(balls, [{"x": .25, "y": .5}])
        self.assertEqual(calls, [.02])
        self.assertEqual(localize_balls(detections, estimator, None, frame, .02), [])

    def test_packet_with_marker_and_ball_is_readable(self):
        packet = {"board_detected": True, "markers": [{"id": 4, "x": .1, "y": .2, "angle": 90}],
                  "balls": [{"x": .3, "y": .4}]}
        text = format_packet(packet, ('127.0.0.1', 5000), 0)
        self.assertIn('ID 4', text)
        self.assertIn('ball x=+0.300m y=+0.400m', text)


if __name__ == '__main__':
    unittest.main()
