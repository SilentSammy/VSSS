import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from vsss import CarState, Field, Pose, WorldState
from vsss.plot import is_close_key, lab_setups, main, make_plotter, marker_styles, render_state
from vsss.simulation import VSSSSim


class PlotSetupsTest(unittest.TestCase):
    def test_car_sprites_stretch_to_fifteen_centimetre_squares(self):
        plotter = make_plotter(width=720)
        for number in range(1, 5):
            style = plotter.marker_overrides[number]
            self.assertEqual(style['size_m'], .15)
            self.assertTrue(style['stretch'])
            self.assertIsInstance(style['image'], np.ndarray)
        # On a blank background the rendered sprite must occupy a .15m square
        # even though its source is rectangular and marker calibration is .09m.
        plotter.background[:] = 255
        image = render_state(plotter, WorldState(cars=[CarState(1, Pose(0, 0, 90))]))
        ys, xs = np.where(np.any(image != 255, axis=2))
        for extent in (xs.max()-xs.min(), ys.max()-ys.min()):
            self.assertAlmostEqual(extent / plotter.scale, .15, delta=.01)

    def test_escape_closes_plot_but_teleop_q_does_not(self):
        self.assertTrue(is_close_key(27))
        self.assertFalse(is_close_key(ord('q')))
        self.assertFalse(is_close_key(ord('Q')))

    def test_catalog_and_default_match_lab_source(self):
        from aruco_gps.vision_setup import DEFAULT_SETUP, VISION_SETUPS
        setups, default = lab_setups()
        self.assertIs(setups, VISION_SETUPS)
        self.assertIs(setups[default], DEFAULT_SETUP)
        plotter = make_plotter(width=200)
        self.assertEqual((plotter.board_width, plotter.board_height), DEFAULT_SETUP.board.get_board_dimensions())
        self.assertEqual((plotter.print_width, plotter.print_height), DEFAULT_SETUP.board.get_print_dimensions())

    def test_every_setup_matches_geometry_marker_size_and_local_artwork(self):
        setups, _ = lab_setups()
        with patch('aruco_gps.cam_config.Camera.open_capture', side_effect=AssertionError('No camera')):
            for name, setup in setups.items():
                with self.subTest(name=name):
                    plotter = make_plotter(name, width=320)
                    self.assertEqual((plotter.board_width, plotter.board_height), setup.board.get_board_dimensions())
                    self.assertEqual((plotter.print_width, plotter.print_height), setup.board.get_print_dimensions())
                    self.assertEqual(plotter.board_config.get_print_dimensions(), setup.board.get_print_dimensions())
                    self.assertEqual(plotter.marker_size_m, setup.marker_size_m)
                    self.assertEqual(plotter.marker_overrides[0]['text'], '6')
                    self.assertEqual(plotter.marker_overrides[0]['color'], [255, 180, 220])
                    self.assertIsInstance(plotter.marker_overrides[8]['image'], np.ndarray)
                    self.assertIsInstance(plotter.marker_overrides[9]['image'], np.ndarray)
                    # All three styles actually render, including image alpha channels.
                    state = WorldState(field=Field(*setup.board.get_print_dimensions()), cars=[
                        CarState(0, Pose(-.04, 0)), CarState(8), CarState(9, Pose(.04, 0))])
                    frame = render_state(plotter, state)
                    self.assertEqual(frame.shape[1], 320)
                    self.assertGreater(frame.std(), 1)

    def test_custom_config_relative_images_and_height_is_not_artwork(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            sprite = np.full((12, 15, 4), (40, 100, 200, 128), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(directory / 'sprite.png'), sprite))
            config = directory / 'markers.json'
            config.write_text(json.dumps({'7': {'image': 'sprite.png', 'scale': 2, 'height_m': .123}}))
            plotter = make_plotter(marker_config=config, width=200)
            self.assertEqual(plotter.marker_overrides[7]['scale'], 2)
            self.assertNotIn('height_m', plotter.marker_overrides[7])
            np.testing.assert_array_equal(plotter.marker_overrides[7]['image'], sprite)
            self.assertNotIn(8, plotter.marker_overrides)  # Custom file replaces default file.

    def test_cli_listing_has_no_provider_and_invalid_selection_fails(self):
        with patch('vsss.plot.ProviderFeed', side_effect=AssertionError('No provider')):
            output = io.StringIO()
            with contextlib.redirect_stdout(output): main(['--list-setups'])
            self.assertIn('setup_letter', output.getvalue())
            self.assertIn('setup_90cm', output.getvalue())
            self.assertIn('setup_160cm: 1.6 x 3 m (default)', output.getvalue())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(['--setup', 'does_not_exist'])

    def test_invalid_styles_fail_at_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'markers.json'
            for style in ({'scale': -1}, {'color': None}, {'text': 'non-ascii: \u2603'}, {'image': 'missing.png'}):
                config.write_text(json.dumps({'1': style}))
                with self.assertRaises((ValueError, OSError)):
                    marker_styles(config)

    def test_cli_selected_setup_is_not_replaced_by_provider_geometry(self):
        with tempfile.TemporaryDirectory() as directory, VSSSSim().start() as sim:
            endpoint = Path(directory) / 'rpc.json'
            output = Path(directory) / '90cm.png'
            with sim.serve(endpoint=endpoint, port=0):
                result = subprocess.run([sys.executable, '-m', 'vsss.plot', '--setup', 'setup_90cm',
                                         '--endpoint', str(endpoint), '--duration', '1', '--width', '320',
                                         '--output', str(output)], capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                image = cv2.imread(str(output))
                expected = make_plotter('setup_90cm', width=320)
                self.assertEqual(image.shape[:2], (expected.height, expected.width))
