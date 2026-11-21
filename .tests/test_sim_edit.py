"""Bag initialization, editor ownership, and mouse-coordinate behavior."""
import json
import math
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

import cv2

from tools._plot_sim import EditTransport, SimDrag
from tools.vsss_plot import plotter_from_snapshot
from vsss._sim import SimWorld, load_config


class SimBagTests(unittest.TestCase):
    def bag(self, rows):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / 'start.jsonl'
        path.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
        return path

    def test_header_and_first_valid_frame_seed_world_and_preserve_visual_metadata(self):
        setup = dict(name='recorded field', marker_size_m=.08, plot_margin_m=.1,
                     marker_overrides={'4': {'color': [30, 50, 70]}},
                     board=dict(dimensions_m=[2., 4.], print_dimensions_m=[2., 4.]),
                     ball_detector=dict(radius_m=.03))
        packet = dict(board_detected=True, markers=[dict(id=4, x=.3, y=.4, angle=20)],
                      balls=[dict(x=-.2, y=-.2)])
        path = self.bag([dict(type='bag_header', schema=3, provider=dict(vision=dict(metadata=dict(status=dict(setup=setup))))),
                         dict(type='udp', valid=False),
                         dict(type='udp', valid=True, packet=dict(board_detected=False)),
                         dict(type='udp', valid=True, packet=packet)])
        # No timing fields required; a corrupt later recording is irrelevant.
        with path.open('a') as output:
            output.write('\n{broken tail')
        world = SimWorld(load_config(bag=path))
        self.assertEqual((world.width, world.height, world.radius), (2, 4, .03))
        self.assertEqual(world.packet()['markers'], packet['markers'])
        self.assertEqual(world.packet()['balls'], packet['balls'])
        self.assertEqual(world.setup()['marker_overrides'], setup['marker_overrides'])
        self.assertEqual(world.setup()['name'], 'recorded field')

    def test_legacy_snapshot_and_config_override_precedence(self):
        packet = dict(board_detected=True, markers=[dict(id=9, x=0, y=0, angle=0)], balls=[])
        bag = self.bag([dict(type='provider_snapshot', vision={}),
                        dict(type='udp', valid=True, packet=packet)])
        config_path = bag.parent / 'config.json'
        config_path.write_text(json.dumps(dict(cars=[dict(id=2, x=.1, y=0, angle=90)])))
        original = SimWorld(load_config(bag=bag))
        self.assertEqual(list(original.cars), [9])
        self.assertEqual(original.balls, [])
        self.assertEqual(original.width, 1.6)
        override = SimWorld(load_config(config_path, bag=bag))
        self.assertEqual(list(override.cars), [2])

    def test_simulator_recording_restores_shape_model_and_first_frame_not_old_initial_pose(self):
        world = SimWorld(load_config())
        world.cars[1].x = .3
        setup = world.setup()
        setup['simulation']['config']['body'] = [[-.08, -.04], [.08, -.04], [0, .08]]
        setup['simulation']['config']['ball_model']['impact_gain'] = 1.1
        bag = self.bag([dict(type='bag_header', provider=dict(vision=dict(metadata=dict(status=dict(setup=setup))))),
                        dict(type='udp', valid=True, packet=world.packet())])
        restored = SimWorld(load_config(bag=bag))
        self.assertEqual(len(restored.body), 3)
        self.assertEqual(restored.seed, world.seed)
        self.assertEqual(restored.model.impact_gain, 1.1)
        self.assertEqual(restored.cars[1].x, .3)
        self.assertNotIn('simulation', restored.config['vision_setup'])

    def test_missing_frame_or_cars_produces_useful_error(self):
        for rows, error in [([], 'no valid frame'),
                            ([dict(type='udp', valid=True, packet=dict(board_detected=True))], 'no cars')]:
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, error):
                load_config(bag=self.bag(rows))


class SimEditTests(unittest.TestCase):
    def setUp(self):
        self.world = SimWorld(load_config())

    def edit(self, operation, kind='car', number=1, **pose):
        return self.world.edit('editor', dict(operation=operation, kind=kind, id=number, **pose))

    def test_car_edit_cancels_waypoint_stops_velocity_and_clamps_rotated_body(self):
        car = self.world.cars[1]
        self.world.waypoint(car, (.3, .5, 90))
        car.vx = .3
        state = self.edit('move', x=4, y=-4, angle=45)
        self.assertEqual((car.mode, car.command, car.vx), (0, (0, 0, 0), 0))
        self.assertIsNone(car.navigator.target)
        self.assertEqual(state['angle'], 45)
        self.assertTrue(all(abs(x) <= .8 + 1e-9 and abs(y) <= 1.5 + 1e-9 for x, y in self.world.vertices(car)))

    def test_grabbed_car_is_anchored_and_disconnect_releases_it(self):
        self.edit('grab')
        self.edit('move', x=0, y=0, angle=0)
        self.world.cars[1].command = (1, 0, 0)
        self.world.balls[0].x = .075  # Overlap resolves against the anchored car.
        for _ in range(20):
            self.world.step(.005)
        self.assertEqual((self.world.cars[1].x, self.world.cars[1].y), (0, 0))
        self.assertGreaterEqual(self.world.balls[0].x, .09 - 1e-9)
        self.assertEqual(len(self.world.release_editor('editor')), 1)
        self.assertFalse(self.world.held)
        self.assertEqual(self.world.cars[1].command, (0, 0, 0))

    def test_ball_edit_is_stationary_exclusive_and_released_without_launch(self):
        self.edit('grab', 'ball', 0)
        self.edit('move', 'ball', 0, x=.4, y=.5)
        self.world.balls[0].vx = 1
        self.world.step(.05)
        self.assertEqual(self.world.balls[0].x, .4)
        with self.assertRaisesRegex(ValueError, 'another editor'):
            self.world.edit('other', dict(operation='move', kind='ball', id=0, x=0, y=0))
        self.edit('release', 'ball', 0)
        self.assertEqual(self.world.balls[0].vx, 0)

    def test_invalid_edits_are_atomic(self):
        original = self.world.object_state('car', 1)
        for values in [dict(x=float('nan'), y=0), dict(x=0, y=0, angle=float('inf'))]:
            with self.assertRaises(ValueError):
                self.edit('move', **values)
            self.assertEqual(self.world.object_state('car', 1), original)
        with self.assertRaises(ValueError):
            self.edit('move', 'ball', -1, x=0, y=0)
        with self.assertRaisesRegex(ValueError, 'not held'):
            self.edit('release')


class SimMouseTests(unittest.TestCase):
    def setUp(self):
        world = SimWorld(load_config())
        self.plotter = plotter_from_snapshot(dict(vision=dict(metadata=dict(status=dict(setup=world.setup())))), 720)
        self.messages = []
        self.drag = SimDrag(self.plotter, [list(v) for v in world.body], world.radius, self.messages.append)
        self.packet = dict(board_detected=True, markers=[dict(id=7, x=.2, y=.3, angle=90)],
                           balls=[dict(x=-.2, y=-.3)])
        self.drag.render(self.packet)

    def mouse(self, event, x, y, flags=0):
        self.drag.on_mouse(event, *self.drag.pixels(x, y), flags)

    def test_car_center_moves_preserving_click_offset_and_heading(self):
        self.mouse(cv2.EVENT_LBUTTONDOWN, .21, .3)
        self.mouse(cv2.EVENT_MOUSEMOVE, .41, .5, cv2.EVENT_FLAG_LBUTTON)
        self.mouse(cv2.EVENT_LBUTTONUP, .41, .5)
        self.assertEqual(self.messages[0], dict(operation='grab', kind='car', id=7))
        self.assertAlmostEqual(self.messages[1]['x'], .4, delta=.005)
        self.assertAlmostEqual(self.messages[1]['y'], .5, delta=.005)
        self.assertEqual(self.messages[1]['angle'], 90)
        self.assertEqual(self.messages[-1]['operation'], 'release')

    def test_car_edge_rotates_in_place_with_correct_cartesian_direction(self):
        self.mouse(cv2.EVENT_LBUTTONDOWN, .26, .3)
        self.mouse(cv2.EVENT_MOUSEMOVE, .2, .36, cv2.EVENT_FLAG_LBUTTON)
        move = self.messages[-1]
        self.assertEqual((move['x'], move['y']), (.2, .3))
        self.assertAlmostEqual(move['angle'], 180, delta=4)

    def test_ball_drag_lost_button_and_empty_canvas(self):
        self.mouse(cv2.EVENT_LBUTTONDOWN, -.2, -.3)
        self.mouse(cv2.EVENT_MOUSEMOVE, -.4, -.5)  # Button released outside canvas.
        self.assertEqual(self.messages[0]['kind'], 'ball')
        self.assertAlmostEqual(self.messages[1]['x'], -.4, delta=.005)
        self.assertEqual(self.messages[-1]['operation'], 'release')
        self.assertIsNone(self.drag.active)
        before = len(self.messages)
        self.mouse(cv2.EVENT_LBUTTONDOWN, .6, .6)
        self.assertEqual(len(self.messages), before)

    def test_coordinate_conversion_uses_current_renderer_geometry(self):
        for width in (320, 1024):
            plotter = plotter_from_snapshot({}, width, fallback=(2, 4))
            drag = SimDrag(plotter, self.drag.body, .02, self.messages.append)
            pixel = drag.pixels(.4, -.7)
            x, y = drag.world_point(*pixel)
            self.assertAlmostEqual(x, .4, delta=1 / plotter.scale)
            self.assertAlmostEqual(y, -.7, delta=1 / plotter.scale)

    def test_transport_coalesces_moves_without_dropping_final_release(self):
        grabbed, gate, released = Event(), Event(), Event()
        calls = []
        class Client:
            def __init__(self, **kwargs):
                self.source_id = kwargs['source_id']
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                pass
            def request(self, **message):
                calls.append(message)
                if message['operation'] == 'grab':
                    grabbed.set()
                    gate.wait(2)
                if message['operation'] == 'release':
                    released.set()
        with patch('tools._plot_sim.SimClient', Client):
            transport = EditTransport('test-source')
            try:
                transport.send(dict(operation='grab', kind='car', id=1))
                self.assertTrue(grabbed.wait(1))
                for index in range(100):
                    transport.send(dict(operation='move', kind='car', id=1, x=index, y=0))
                transport.send(dict(operation='release', kind='car', id=1))
                gate.set()
                self.assertTrue(released.wait(1))
                self.assertEqual([call['operation'] for call in calls], ['grab', 'move', 'release'])
                self.assertEqual(calls[1]['x'], 99)
            finally:
                gate.set()
                transport.close()


if __name__ == '__main__':
    unittest.main()
