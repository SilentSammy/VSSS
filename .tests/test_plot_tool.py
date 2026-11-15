import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from tools.vsss_plot import plotter_from_snapshot, main


class PlotToolTests(unittest.TestCase):
    def test_renderer_places_marker_and_ball_on_field(self):
        plotter = plotter_from_snapshot({}, width=320, fallback=(1.6, 3.0))
        frame = plotter.render_packet({'board_detected': True,
                                'markers': [{'id': 1, 'x': 0, 'y': .5, 'angle': 90}],
                                'balls': [{'x': .2, 'y': -.3}]})
        self.assertEqual(frame.shape[1], 320)
        self.assertGreater(frame.shape[0], frame.shape[1])
        self.assertFalse((frame == plotter.background).all())

    def test_headless_bag_replay_saves_last_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            bag = Path(directory) / 'bag.jsonl'
            output = Path(directory) / 'field.png'
            rows = [
                {'type': 'provider_snapshot', 'monotonic_ns': 1,
                 'controlled_cars': [], 'vision': {}},
                {'type': 'udp', 'monotonic_ns': 10, 'valid': True,
                 'packet': {'board_detected': True,
                            'markers': [{'id': 1, 'x': .1, 'y': .2, 'angle': 45}],
                            'balls': [{'x': -.2, 'y': .3}]}}
            ]
            bag.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                main(['--bag', str(bag), '--speed', '1000', '--width', '320',
                      '--field-width', '1.6', '--field-height', '3.0',
                      '--output', str(output)])
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1000)
            self.assertIn('Saved', stdout.getvalue())


if __name__ == '__main__':
    unittest.main()
