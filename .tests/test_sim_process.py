"""Real provider/client/tool subprocesses; hardware paths are forbidden."""
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vsss import Car, GPS, gps
from vsss._ble_client import MecanumBLEClient
from vsss._bag import BagReader
from vsss.provider import ProviderClient, available_cars
from vsss.sim import SimClient

ROOT = Path(__file__).resolve().parents[1]
GUARDED_SIM = '''
from unittest.mock import patch
from tools.vsss_sim import main
with patch('vsss._provider.VisionSource', side_effect=AssertionError('physical UDP source')), \\
     patch('vsss._provider.Provider.available_cars', side_effect=AssertionError('BLE scan')), \\
     patch('vsss._provider_car.subprocess.Popen', side_effect=AssertionError('BLE worker')):
    main()
'''


class SimProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        self.env = patch.dict(os.environ, {'VSSS_PROVIDER_PORT': str(port)})
        self.env.start()
        self.token_path = Path(tempfile.gettempdir()) / f'vsss-provider-{port}.token'
        self.clients = []
        self.process = subprocess.Popen([sys.executable, '-u', '-c', GUARDED_SIM, '--cars', '1', '2'],
                                        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.fail(self.process.communicate()[1])
            if self.token_path.exists() and self.token_path.read_text():
                break
            time.sleep(.01)
        else:
            self.fail('simulation provider did not start')

    def tearDown(self):
        for client in self.clients:
            client.disconnect()
        Car.close_all()
        gps.close()
        self.process.terminate()
        self.process.communicate(timeout=5)
        # Forced termination cannot run the provider's token cleanup.
        self.token_path.unlink(missing_ok=True)
        self.env.stop()
        self.temp.cleanup()

    def client(self, name='car1'):
        client = MecanumBLEClient(name)
        self.clients.append(client)
        client.connect()
        client.wait_until_connected(3)
        return client

    def test_discovery_public_car_gps_and_configuration(self):
        self.assertEqual(available_cars(refresh=True), [1, 2])
        with Car(1) as car:
            pose = car.wait_for_position(2)
            self.assertAlmostEqual(pose.y, -.45, delta=.01)
            car.set_velocity(x=.5)
            time.sleep(.4)  # Allow command delivery, acceleration, then delayed feedback.
            self.assertGreater(car.read_status().y, pose.y + .03)
            car.stop()
            self.assertEqual(car.read_status().mode_name, 'idle')
            matrix = ((.9, .1, 0), (0, 1, 0), (0, 0, 1))
            self.assertEqual(car.set_trim(matrix), matrix)
            self.assertEqual(car.read_trim(), matrix)
            self.assertEqual(car.set_controller({'max_speed': .5})['max_speed'], .5)
            self.assertEqual(car.read_controller()['max_speed'], .5)
            self.assertTrue(car.read_status().pose_fresh)
            self.assertEqual(len(GPS().balls), 1)

    def test_editor_ipc_moves_holds_and_disconnects_without_stale_motion(self):
        car = self.client()
        with ProviderClient(('udp',)) as feed:
            source_id = feed.snapshot['vision']['simulation_control']['source_id']
            with SimClient(source_id=source_id) as editor, SimClient() as other:
                editor.grab('car', 1)
                editor.move_car(1, .3, .4, 30)
                editor.move_ball(0, -.3, -.2)
                car.set_velocity({'x': .7})
                time.sleep(.15)
                status = car.read_status()
                self.assertAlmostEqual(status.x, .3)
                self.assertAlmostEqual(status.y, .4)
                self.assertEqual(status.angle, 30)
                with self.assertRaisesRegex(ValueError, 'another editor'):
                    other.move_car(1, 0, 0)
                with self.assertRaises(ValueError):
                    editor.move_ball(-1, 0, 0)
                self.assertTrue(editor.state()['cars'][0]['held'])
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline:
                    event = feed.read(.1)
                    if event and event['type'] == 'udp' and event['packet']['markers'][0]['x'] == .3:
                        break
                self.assertAlmostEqual(event['packet']['markers'][0]['x'], .3)
            with SimClient() as editor:
                deadline = time.monotonic() + 1
                while editor.state()['cars'][0]['held'] and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertFalse(editor.state()['cars'][0]['held'])
            self.assertEqual(car.read_status().mode_name, 'idle')

    def test_editor_binds_to_feed_and_recovers_grab_after_abrupt_process_exit(self):
        with self.assertRaisesRegex(ConnectionError, 'instance changed'):
            SimClient(source_id='old-instance')
        result = subprocess.run([sys.executable, '-c',
            "from vsss.sim import SimClient; import os; c=SimClient(); c.grab('ball',0); c.move_ball(0,.4,.5); os._exit(0)"],
            cwd=ROOT, capture_output=True, text=True, timeout=4)
        self.assertEqual(result.returncode, 0, result.stderr)
        with SimClient() as client:
            deadline = time.monotonic() + 1
            state = client.state()
            while state['balls'][0]['held'] and time.monotonic() < deadline:
                time.sleep(.01)
                state = client.state()
            self.assertFalse(state['balls'][0]['held'])
            self.assertEqual((state['balls'][0]['x'], state['balls'][0]['y']), (.4, .5))

    def test_mouse_gesture_reaches_simulator_through_background_transport(self):
        import cv2
        from tools._plot_sim import EditTransport, SimDrag
        from tools.vsss_plot import plotter_from_snapshot
        with ProviderClient(('udp',)) as feed:
            plotter = plotter_from_snapshot(feed.snapshot, 720)
            setup = feed.snapshot['vision']['metadata']['status']['setup']
            transport = EditTransport(feed.snapshot['vision']['source_id'])
            try:
                drag = SimDrag(plotter, setup['simulation']['body'], .02, transport.send)
                drag.render(feed.read(1)['packet'])
                drag.on_mouse(cv2.EVENT_LBUTTONDOWN, *drag.pixels(-.11, -.45), 0)
                drag.on_mouse(cv2.EVENT_MOUSEMOVE, *drag.pixels(-.4, .4), cv2.EVENT_FLAG_LBUTTON)
                drag.on_mouse(cv2.EVENT_LBUTTONUP, *drag.pixels(-.4, .4), 0)
                with SimClient() as client:
                    deadline = time.monotonic() + 2
                    while time.monotonic() < deadline:
                        car = client.state()['cars'][0]
                        if abs(car['x'] + .4) < .005 and not car['held']:
                            break
                        time.sleep(.01)
                self.assertAlmostEqual(car['x'], -.4, delta=.005)
                self.assertAlmostEqual(car['y'], .4, delta=.005)
                self.assertEqual(car['angle'], 90)
                self.assertFalse(car['held'])
            finally:
                transport.close()

    def test_cli_initializes_from_first_bag_frame_and_header(self):
        bag = self.directory / 'initial.jsonl'
        setup = dict(board=dict(dimensions_m=[2, 4], print_dimensions_m=[2, 4]),
                     marker_size_m=.08, ball_detector=dict(radius_m=.025))
        rows = [dict(type='bag_header', provider=dict(vision=dict(metadata=dict(status=dict(setup=setup))))),
                dict(type='udp', valid=True, packet=dict(board_detected=True,
                    markers=[dict(id=8, x=.2, y=.3, angle=45)], balls=[dict(x=-.2, y=.5)]))]
        bag.write_text('\n'.join(json.dumps(row) for row in rows))
        token = self.token_path.read_text()
        self.process.terminate()
        self.process.communicate(timeout=5)
        self.process = subprocess.Popen([sys.executable, '-u', '-c', GUARDED_SIM, str(bag), '--seed', '123'],
                                        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + 3
        while self.token_path.read_text() == token and time.monotonic() < deadline:
            self.assertIsNone(self.process.poll())
            time.sleep(.01)
        self.assertEqual(available_cars(), [8])
        with ProviderClient(('udp',)) as feed:
            self.assertEqual(feed.snapshot['vision']['metadata']['status']['setup']['board']['dimensions_m'], [2, 4])
            self.assertEqual(feed.snapshot['vision']['metadata']['status']['setup']['simulation']['config']['random_seed'], 123)
            event = feed.read(1)
            self.assertEqual(event['packet']['markers'], rows[1]['packet']['markers'])

    def test_shared_motion_ownership_and_owner_departure_stop(self):
        first, second, reader = self.client(), self.client(), self.client()
        first.set_velocity({'x': .6})
        time.sleep(.2)
        second.set_velocity({'x': -.5})
        time.sleep(.2)
        first.disconnect()  # No longer owns motion.
        self.assertEqual(reader.read_status().mode_name, 'manual')
        before = reader.read_status().y
        time.sleep(.2)
        self.assertLess(reader.read_status().y, before)
        second.disconnect()
        time.sleep(.15)  # Flush pre-stop UDP feedback before checking the held pose.
        stopped = reader.read_status()
        self.assertEqual(stopped.mode_name, 'idle')
        time.sleep(.2)
        self.assertEqual(reader.read_status().y, stopped.y)

    def test_udp_and_ble_transmission_latency_are_visible_to_clients(self):
        with ProviderClient(('udp',)) as feed:
            packet = feed.read(1)
            self.assertGreaterEqual(packet['packet']['frame_age_ms'], 95)
            age = packet['monotonic_ns'] / 1e9 - packet['packet']['timestamp']
            self.assertAlmostEqual(age, packet['packet']['frame_age_ms'] / 1000, delta=.02)
        client = self.client()
        before = time.perf_counter()
        client.set_waypoint(-.11, -.25, 90)
        self.assertGreaterEqual(time.perf_counter() - before, .035)

    def test_student_example_moves_ball_into_zone_with_latency_and_wobble(self):
        result = subprocess.run([sys.executable, str(ROOT / 'ball_to_zone.py'),
                                 '--target', '0', '.3', '--radius', '.18', '--timeout', '30'],
                                cwd=ROOT, capture_output=True, text=True, timeout=38)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('Ball settled in the zone', result.stdout)
        with SimClient() as editor:
            ball = editor.state()['balls'][0]
            self.assertLessEqual(((ball['x']) ** 2 + (ball['y'] - .3) ** 2) ** .5, .18)
            time.sleep(.2)
            after = editor.state()['balls'][0]
            self.assertLess(((after['x'] - ball['x']) ** 2 + (after['y'] - ball['y']) ** 2) ** .5, .008)

    def test_waypoint_completion_velocity_override_and_restart(self):
        car, reader = self.client(), self.client()
        car.set_waypoint(-.11, -.25, 90)
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            status = reader.read_status()
            if status.mode_name == 'reached':
                break
            time.sleep(.05)
        self.assertEqual(status.mode_name, 'reached')
        self.assertAlmostEqual(status.y, -.25, delta=.025)
        car.set_waypoint(-.11, .5, 90)
        reader.set_velocity({'y': -.3})
        time.sleep(.1)
        self.assertEqual(car.read_status().mode_name, 'manual')
        car.restart_robot(timeout=4)
        self.assertEqual(car.read_status().mode_name, 'idle')
        self.assertTrue(reader.is_connected)

    def test_multiple_cars_and_unknown_id(self):
        first, second = self.client(), self.client('car2')
        first.set_velocity({'x': .5})
        time.sleep(.2)
        self.assertGreater(first.read_status().y, second.read_status().y)
        with self.assertRaises(TimeoutError):
            self.client('car99')
        self.assertTrue(first.is_connected)

    def test_invalid_control_requests_report_errors_without_taking_motion(self):
        owner, other = self.client(), self.client()
        owner.set_velocity({'x': .3})
        with self.assertRaises(RuntimeError):
            other._operation('write_controller', values=[float('nan')] * 15)
        with self.assertRaises(RuntimeError):
            other._operation('write_trim', values=[1, 2])
        with self.assertRaises(RuntimeError):
            other._operation('unsupported')
        self.assertEqual(owner.read_status().mode_name, 'manual')
        self.assertEqual(owner.read_controller()['max_speed'], .4)
        other.disconnect()
        self.assertEqual(owner.read_status().mode_name, 'manual')

    def test_observation_fanout_raw_payloads_and_simulated_control_events(self):
        with ProviderClient() as first, ProviderClient() as second:
            self.assertTrue(first.snapshot['vision']['simulated'])
            self.assertEqual(first.snapshot['vision']['metadata']['status']['setup']['board']['dimensions_m'], [1.6, 3.])
            car = self.client()
            car.set_velocity({'x': .4})
            car.stop()
            udp, ble, owner = {}, [], []
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and (len(udp) < 3 or not ble or not owner):
                event = first.read(.1)
                if event is None:
                    continue
                if event['type'] == 'udp':
                    self.assertEqual(json.loads(base64.b64decode(event['payload_base64'])), event['packet'])
                    udp[event['packet']['frame_sequence']] = event
                elif event['type'] == 'ble':
                    ble.append(event)
                    self.assertTrue(event['simulated'])
                    self.assertEqual(event['session_id'], event['origin_session_id'])
                elif event['type'] == 'motion_owner':
                    owner.append(event)
            self.assertTrue(ble)
            self.assertTrue(owner)
            matched = False
            for _ in range(50):
                event = second.read(.1)
                if event and event['type'] == 'udp' and event['packet']['frame_sequence'] in udp:
                    self.assertEqual(event, udp[event['packet']['frame_sequence']])
                    matched = True
                    break
            self.assertTrue(matched)

    def test_observer_recording_bag_replay_and_plot_subprocesses(self):
        bag = self.directory / 'simulation.jsonl'
        result = subprocess.run([sys.executable, '-m', 'tools.vsss_observer', str(bag),
                                 '--duration', '.3'], cwd=ROOT, capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [json.loads(line) for line in bag.read_text().splitlines()]
        self.assertEqual(rows[0]['schema'], 3)
        self.assertTrue(rows[-1]['complete'])
        with BagReader(bag, speed=1000) as reader:
            self.assertTrue(reader.snapshot['vision']['simulated'])
            self.assertEqual(reader.read(1)['type'], 'udp')
        image = self.directory / 'field.png'
        result = subprocess.run([sys.executable, '-m', 'tools.vsss_plot', '--duration', '.3',
                                 '--width', '320', '--output', str(image)],
                                cwd=ROOT, capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreater(image.stat().st_size, 1000)
        token = self.token_path.read_text()
        self.process.terminate()
        self.process.communicate(timeout=5)
        self.process = subprocess.Popen([sys.executable, '-m', 'tools.vsss_bag', str(bag), '--speed', '2'],
                                        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + 3
        while self.token_path.read_text() == token and time.monotonic() < deadline:
            self.assertIsNone(self.process.poll())
            time.sleep(.01)
        with self.assertRaisesRegex(ConnectionError, 'does not support'):
            SimClient()
        replay_image = self.directory / 'replay.png'
        result = subprocess.run([sys.executable, '-m', 'tools.vsss_plot', '--width', '320',
                                 '--output', str(replay_image)], cwd=ROOT,
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(replay_image.is_file())
        self.assertEqual(self.process.wait(timeout=3), 0)

    def test_teleop_input_drives_real_simulation_client(self):
        import vsss
        from tools import vsss_teleop
        inp = SimpleNamespace(KEYBOARD_AVAILABLE=True, pressed_keys=set(),
                              get_bipolar_ctrl=lambda *args, **kwargs: 1 if args == ('w', 's') else 0)
        from tools.vsss_teleop import velocity as real_velocity
        deadline = time.monotonic() + .4
        def command(methods, state):
            if time.monotonic() >= deadline:
                raise KeyboardInterrupt
            return real_velocity(methods, state)
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.dict(sys.modules, {'vsss.input': inp, 'inputs': SimpleNamespace(devices=SimpleNamespace(gamepads=[]))}), \
                patch.object(vsss, 'input', inp, create=True), \
                patch.object(vsss_teleop, 'velocity', side_effect=command):
            vsss_teleop.main(['--car', '1=wasd'])
        car = self.client()
        self.assertGreater(car.read_status().y, -.43)
        self.assertEqual(car.read_status().mode_name, 'idle')

    def test_refuses_competing_providers_and_stays_selected_without_clients(self):
        token = self.token_path.read_text()
        for module in ('tools.vsss_live', 'tools.vsss_sim'):
            result = subprocess.run([sys.executable, '-m', module], cwd=ROOT,
                                    capture_output=True, text=True, timeout=4)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('already be active', result.stderr)
            self.assertEqual(self.token_path.read_text(), token)
        time.sleep(.3)
        self.assertIsNone(self.process.poll())


if __name__ == '__main__':
    unittest.main()
