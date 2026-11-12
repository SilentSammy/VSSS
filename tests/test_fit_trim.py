import unittest

from tools.fit_trim import fit


class TrimFitTests(unittest.TestCase):
    def test_fit_composes_with_existing_trim(self):
        response = ((1, .2, 0), (0, 1, 0), (0, 0, 2))
        rows = []
        for axis in range(3):
            for sign in (-1, 1):
                command = [0., 0., 0.]
                command[axis] = sign * .5
                measured = [response[i][axis] * command[axis] for i in range(3)]
                rows.append(dict(zip(('command_x', 'command_y', 'command_w',
                                      'forward_m', 'left_m', 'heading_deg'),
                                     command + measured)))
        result = fit(rows, ((1.1, 0, 0), (0, 1, 0), (0, 0, 1)))
        self.assertAlmostEqual(result['matrix'][0][0], 1.1)
        self.assertAlmostEqual(result['matrix'][0][1], -.22)
        self.assertEqual(result['samples'], 6)

    def test_requires_both_pulse_directions(self):
        with self.assertRaisesRegex(ValueError, 'positive and negative'):
            fit([], ((1, 0, 0), (0, 1, 0), (0, 0, 1)))

    def test_different_pulse_durations_fit_same_velocity_response(self):
        response = ((1, .2, .1), (0, 1, 0), (.3, .1, 2))
        rows = []
        for axis, duration in enumerate((.8, .8, .55)):
            for sign in (-1, 1):
                command = [0., 0., 0.]
                command[axis] = sign * .5
                measured = [response[i][axis] * command[axis] * duration for i in range(3)]
                row = dict(zip(('command_x', 'command_y', 'command_w',
                                'forward_m', 'left_m', 'heading_deg'), command + measured))
                row['duration_s'] = duration
                rows.append(row)
        result = fit(rows, ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        for actual, expected in zip(result['response_matrix'], response):
            for a, b in zip(actual, expected):
                self.assertAlmostEqual(a, b)
        rows[0]['duration_s'] = 0
        with self.assertRaisesRegex(ValueError, 'duration'):
            fit(rows, ((1, 0, 0), (0, 1, 0), (0, 0, 1)))


if __name__ == '__main__':
    unittest.main()
