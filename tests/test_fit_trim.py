import unittest

from experiments.fit_trim import fit


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


if __name__ == '__main__':
    unittest.main()
