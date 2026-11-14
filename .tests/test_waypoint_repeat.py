"""Oscillation classification independent of physical hardware."""
import unittest
from tools.waypoint_repeat import oscillating


class OscillationTests(unittest.TestCase):
    def test_repeated_crossings_are_flagged(self):
        samples = [(i * .1, 0., .4 + (.05 if (i // 7) % 2 else -.05), 90.) for i in range(60)]
        self.assertTrue(oscillating(samples, (0., .4, 90.)))

    def test_camera_noise_and_a_normal_approach_are_not_flagged(self):
        noise = [(i * .1, 0., .4 + (.01 if i % 2 else -.01), 90.) for i in range(60)]
        approach = [(i * .1, 0., -.4 + i * .8 / 59, 90.) for i in range(60)]
        self.assertFalse(oscillating(noise, (0., .4, 90.)))
        self.assertFalse(oscillating(approach, (0., .4, 90.)))

    def test_heading_wraparound_is_not_an_oscillation(self):
        samples = [(i * .1, 0., .4, -179. if i % 2 else 179.) for i in range(60)]
        self.assertFalse(oscillating(samples, (0., .4, 180.)))


if __name__ == '__main__':
    unittest.main()
