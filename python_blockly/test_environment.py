"""Real local RPC integration for editor provider selection and embedded edits."""
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from environment import Environment
from vsss import VSSSProvider, EventHub, GPSState, CarCmd, CarState, Pose, StatePatch
from vsss.plot import EmbeddedPlot
from vsss.simulation import VSSSSim


def wait_for(predicate):
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        if predicate(): return
        time.sleep(.02)
    raise AssertionError('timed out')


class EnvironmentTest(unittest.TestCase):
    def test_owned_live_and_sim_modes_switch_with_cleanup_without_hardware(self):
        class Cars(CarCmd):
            def available(self, refresh=False): return []
            def _apply(self, number, value): pass
        with tempfile.TemporaryDirectory() as directory:
            lab = Environment(Path(directory)/'rpc.json', port=0)
            with patch('vsss.live.GPSLive', side_effect=lambda events: GPSState(events)), \
                 patch('vsss.live.CarCmdLive', side_effect=Cars):
                try:
                    first = lab.select('goalie_easy')
                    live = lab.select('live')
                    self.assertEqual(live['mode'], 'live')
                    self.assertNotEqual(live['source_id'], first['source_id'])
                    plot = EmbeddedPlot(lab.endpoint, width=200)
                    try:
                        wait_for(lambda: plot.feed.snapshot()[2] == live['source_id'])
                        _, metadata = plot.frame()
                        self.assertFalse(metadata['editable'])
                        with self.assertRaisesRegex(ValueError, 'simulation only'):
                            plot.pointer('down', 100, 100, owner='editor',
                                         source_id=metadata['source_id'], generation=None)
                        current = lab.select('scenario')
                        self.assertEqual(current['mode'], 'simulation')
                        wait_for(lambda: plot.feed.snapshot()[2] == current['source_id'])
                        self.assertTrue(plot.frame()[1]['editable'])
                    finally:
                        plot.close()
                finally:
                    lab.close()

    def test_start_reset_and_keep_current_state_on_run(self):
        with tempfile.TemporaryDirectory() as directory:
            endpoint = Path(directory)/'rpc.json'
            lab = Environment(endpoint, port=0)
            try:
                self.assertIsNone(lab.status()['mode'])
                first = lab.select('goalie_easy')
                self.assertEqual(first['mode'], 'simulation')
                with VSSSProvider.connect(endpoint) as client:
                    self.assertEqual(client.gps.state().field.width, 1.6)
                    client.update_state(StatePatch(cars=[CarState(1, Pose(.3, .3))]))
                    lab.ensure()
                    self.assertAlmostEqual(client.gps.state().cars[0].pose.x, .3)
                    lab.select('goalie_easy')
                    self.assertAlmostEqual(client.gps.state().cars[0].pose.y, -1.1)
                    lab.select('scenario')
                    self.assertEqual(client.describe()['source_id'], first['source_id'])
                    self.assertAlmostEqual(client.gps.state().cars[0].pose.x, -.2)
                with self.assertRaises(ValueError): lab.select('../anything')
            finally:
                lab.close()
            self.assertFalse(endpoint.exists())

    def test_external_simulator_is_reset_and_survives_editor_close(self):
        with tempfile.TemporaryDirectory() as directory, VSSSSim().start() as sim:
            endpoint = Path(directory)/'rpc.json'
            with sim.serve(endpoint=endpoint, port=0):
                lab = Environment(endpoint, port=0)
                try:
                    lab.select('goalie_easy')
                    self.assertAlmostEqual(sim.gps.state().cars[0].pose.y, -1.1)
                    self.assertIsNone(lab.owned)
                    with self.assertRaisesRegex(ValueError, 'another terminal'):
                        lab.select('live')
                finally:
                    lab.close()
                with VSSSProvider.connect(endpoint) as client:
                    self.assertEqual(client.describe()['source_id'], sim.source_id)

    def test_embedded_plot_drag_expiry_and_stale_world_protection(self):
        with tempfile.TemporaryDirectory() as directory, VSSSSim().start() as sim:
            endpoint = Path(directory)/'rpc.json'
            with sim.serve(endpoint=endpoint, port=0):
                plot = EmbeddedPlot(endpoint, width=320)
                try:
                    wait_for(lambda: plot.feed.snapshot()[2] == sim.source_id)
                    frame, metadata = plot.frame()
                    self.assertGreater(frame.std(), 1)
                    args = dict(owner='editor1', source_id=metadata['source_id'], generation=metadata['generation'])
                    x, y = plot.plotter.width/2, plot.plotter.height/2
                    plot.pointer('down', x, y, **args)
                    wait_for(lambda: ('car', 1) in sim._held)
                    plot.pointer('move', x+.2*plot.plotter.scale, y, **args)
                    wait_for(lambda: abs(sim._world.cars[0].pose.x-.2) < .01)
                    with self.assertRaisesRegex(ValueError, 'another editor'):
                        plot.pointer('down', x, y, **dict(args, owner='editor2'))
                    plot.touched = time.monotonic()-2
                    plot.frame()
                    wait_for(lambda: not sim._held)
                    sim.load_scenario(sim.scenario())
                    wait_for(lambda: plot.feed.info()['simulation']['generation'] != metadata['generation'])
                    with self.assertRaisesRegex(ValueError, 'fresh frame'):
                        plot.pointer('down', x, y, **args)
                finally:
                    plot.close()
