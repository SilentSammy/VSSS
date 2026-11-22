import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from vsss import VSSSProvider, Velocity
from vsss.teleop import main as teleop
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

    def test_invalid_speed(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            teleop(['--speed', 'nan'])
