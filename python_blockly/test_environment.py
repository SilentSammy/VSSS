"""Real local RPC integration for editor provider selection and embedded edits."""
from pathlib import Path
import builtins
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
from threading import Lock, Thread
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

from environment import Environment
from vsss import VSSSProvider, EventHub, GPSState, CarCmd, CarState, BallState, Pose, StatePatch, Velocity, WorldState
from vsss.plot import EmbeddedPlot
from vsss.simulation import VSSSSim
from vsss.scenarios import load_scenario


def wait_for(predicate):
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        if predicate(): return
        time.sleep(.02)
    raise AssertionError('timed out')


class EnvironmentTest(unittest.TestCase):
    def test_provider_errors_return_json_and_allow_retry(self):
        from server import Handler
        class ProviderFailure(Exception): pass
        class FailingEnvironment:
            failed = True
            def status(self):
                if self.failed: raise ProviderFailure('provider unreachable')
                return {'options': [], 'selected': 'car1_ball1', 'mode': None}
            def select(self, name): return self.status()
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        server.environment = FailingEnvironment()
        server.runner = SimpleNamespace(status=lambda: {'state': 'idle'})
        server.action_lock = Lock()
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = HTTPConnection('127.0.0.1', server.server_port, timeout=3)
            connection.request('GET', '/api/environment')
            response = connection.getresponse()
            self.assertEqual(response.status, 503)
            self.assertIn('provider unreachable', json.loads(response.read())['error'])
            headers = {'Origin': f'http://127.0.0.1:{server.server_port}', 'X-VSSS-Blockly': '1',
                       'Content-Type': 'application/json'}
            connection.request('POST', '/environment', json.dumps({'selection': 'car1_ball1'}), headers)
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            self.assertIn('provider unreachable', json.loads(response.read())['error'])
            server.environment.failed = False
            connection.request('POST', '/environment', json.dumps({'selection': 'car1_ball1'}), headers)
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read())['selected'], 'car1_ball1')
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)

    def test_catalog_is_local_and_has_the_requested_fleets(self):
        with tempfile.TemporaryDirectory() as directory:
            lab = Environment(Path(directory)/'rpc.json', port=0)
            self.assertEqual(lab.selected, 'car1_ball1')
            self.assertEqual([item['label'] for item in lab.options()], [
                'Live', '1 car', '1 car 1 ball', '2 cars 1 ball', '4 cars 1 ball', 'goalie easy', 'goalie hard'])
            with patch('vsss.live.GPSLive', side_effect=AssertionError('Simulator must not open live UDP')):
                try:
                    for name, car_count, ball_count in [('car1',1,0),('car1_ball1',1,1),
                                                        ('cars2_ball1',2,1),('cars4_ball1',4,1)]:
                        lab.select(name)
                        with VSSSProvider.connect(lab.endpoint) as client:
                            state = client.gps.state()
                            self.assertEqual((len(state.cars),len(state.balls)), (car_count,ball_count))
                finally: lab.close()

    def test_waypoint_goalie_corrects_heading_and_holds_lane(self):
        root = Path(__file__).resolve().parent
        program = (root / 'examples/09_goalie.py').read_text()

        class ShotEnded(BaseException):
            pass

        def follows(heading):
            state = WorldState(cars=[CarState(1,Pose(.15,1.2,heading))],balls=[BallState(0,Pose(.3,-.8))])
            with VSSSSim(state) as sim:
                ticks = 0
                waypoints = []
                class Car:
                    def __init__(self, number): self.number = number
                    @property
                    def pose(self): return sim._world.cars[0].pose
                    def wait_for_position(self): return self.pose
                    def go_to(self, x, y, angle):
                        waypoints.append((x,y,angle))
                        sim.cars.go_to(self.number,Pose(x,y,angle))
                    def stop(self): sim.cars.stop(self.number)
                class GPS:
                    @property
                    def ball(self): return sim._world.balls[0].pose
                def sleep(dt):
                    nonlocal ticks
                    for _ in range(round(dt/.01)):
                        sim.step(.01)
                        ticks += 1
                        if ticks >= 600: raise ShotEnded()
                def imports(name, *args, **kwargs):
                    if name == 'vsss': return SimpleNamespace(Car=Car, gps=GPS(), run=lambda fn: fn)
                    if name == 'time': return SimpleNamespace(sleep=sleep)
                    return builtins.__import__(name, *args, **kwargs)
                scope = {'__name__': 'example', '__builtins__': dict(vars(builtins), __import__=imports)}
                exec(program, scope)
                try: scope['main']()
                except ShotEnded: pass
                self.assertGreater(len(waypoints),50)
                self.assertTrue(all(target == (.3,1.2,-90) for target in waypoints))
                pose = sim._world.cars[0].pose
                self.assertAlmostEqual(pose.x,.3,delta=.04)
                self.assertAlmostEqual(pose.y,1.2,delta=.04)
                self.assertLess(abs((pose.angle+90+180)%360-180),5)
        for heading in (0,90,-90):
            with self.subTest(heading=heading): follows(heading)

    def test_goalie_starts_have_motion_and_increasing_difficulty(self):
        root = Path(__file__).resolve().parent.parent / 'examples'
        easy, hard = [load_scenario(root / f'goalie_{level}.jsonl') for level in ('easy', 'hard')]
        for scenario in (easy, hard):
            self.assertTrue(scenario.wait_for_control)
            self.assertGreater(scenario.motion[-1].vy, 0)
            self.assertEqual(scenario.state.cars[0].pose.angle, -90)
            self.assertGreater(scenario.state.cars[0].pose.y, 1)
            self.assertNotEqual(scenario.state.cars[0].pose.x, 0)
            with VSSSSim(scenario) as sim:
                initial = sim._world.balls[0].pose
                sim.step(.1)
                self.assertEqual(sim._world.balls[0].pose, initial)
                sim.cars.set_velocity(1, Velocity())
                sim.step(.1)
                self.assertGreater(sim._world.balls[0].pose.y, initial.y)
        self.assertGreater(abs(hard.motion[-1].vy), abs(easy.motion[-1].vy))
        self.assertLess(hard.state.cars[0].pose.x * hard.state.balls[0].pose.x, 0)

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
                        current = lab.select('car1_ball1')
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
                    self.assertAlmostEqual(client.gps.state().cars[0].pose.y, 1.2)
                    lab.select('car1_ball1')
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
                    self.assertAlmostEqual(sim.gps.state().cars[0].pose.y, 1.2)
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
