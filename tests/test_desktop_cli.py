import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from vsss import VSSSProvider, Velocity
from vsss.desktop import draw_world, key_velocity, teleop
from vsss.simulation import VSSSSim, main, scenario_state


class DesktopCliTest(unittest.TestCase):
    def test_load_running_simulator_keeps_client_and_stops_motion(self):
        scenario = Path(__file__).resolve().parents[1] / 'examples' / 'goalie_easy.jsonl'
        with tempfile.TemporaryDirectory() as directory, VSSSSim() as sim:
            endpoint = Path(directory) / 'rpc.json'
            with sim.serve(endpoint=endpoint, port=0), VSSSProvider.connect(endpoint) as client:
                original = client.describe()['source_id']
                client.cars.set_velocity(1, Velocity(.3))
                with contextlib.redirect_stdout(io.StringIO()):
                    main(['--load', str(scenario), '--endpoint', str(endpoint)])
                self.assertEqual(client.describe()['source_id'], original)
                self.assertEqual(client.gps.state().cars, scenario_state(scenario).cars)
                self.assertEqual(sim._velocities, {})
                sim.step(.1)
                self.assertEqual(client.gps.state().cars, scenario_state(scenario).cars)

    def test_held_keys_and_opposing_keys(self):
        self.assertEqual(key_velocity({'w', 'a', 'q'}, .3), Velocity(.3, .3, .3))
        self.assertEqual(key_velocity({'w', 's', 'a', 'd', 'q', 'e'}, .3), Velocity())
        self.assertEqual(key_velocity(set(), .3), Velocity())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            teleop(['--speed', 'nan'])

    def test_plot_projects_world_coordinates(self):
        class Canvas:
            def winfo_width(self): return 650
            def winfo_height(self): return 800
            def delete(self, *args): pass
            def __getattr__(self, name):
                if name.startswith('create_'):
                    return lambda *args, **kwargs: None
                raise AttributeError(name)
        with VSSSSim() as sim:
            positions = draw_world(Canvas(), sim.gps.state())
            car, ball = positions
            self.assertEqual(car, ('car', 1, 325, 400))
            self.assertGreater(ball[2], car[2])
            self.assertLess(ball[3], car[3])
