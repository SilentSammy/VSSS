"""Real process checks for replay fanout, control isolation, GPS and plotting."""
import base64
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from vsss.provider import ProviderClient, available_cars
from vsss._provider_connection import ProviderConnection
from vsss.gps import GPS

ROOT = Path(__file__).resolve().parents[1]


class BagProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        self.env = patch.dict(os.environ, {'VSSS_PROVIDER_PORT': str(port)})
        self.env.start()
        self.token_path = Path(tempfile.gettempdir()) / f'vsss-provider-{port}.token'
        self.process = None
        self.observers = []
        self.bag = self.directory / 'session.jsonl'
        setup = dict(name='test-field', marker_size_m=.09, plot_margin_m=.05,
                     board=dict(dimensions_m=[1.6, 3.], print_dimensions_m=[1.6, 3.]),
                     ball_detector=dict(radius_m=.02))
        rows = [dict(type='bag_header', schema=3,
            provider=dict(type='provider_snapshot', controlled_cars=[],
                          vision=dict(local_port=5001, metadata=dict(status=dict(setup=setup)))))]
        for index in range(3):
            packet = dict(board_detected=True, timestamp=1000 + index * .4,
                          markers=[dict(id=1, x=.1 * index, y=.75, angle=90)],
                          balls=[dict(x=.2, y=.8)], frame_age_ms=10)
            rows.append(dict(type='udp', valid=True, sequence=index,
                             source_id='original', monotonic_ns=1_000_000_000 + index * 400_000_000,
                             wall_time_ns=1_000_000_000_000 + index * 400_000_000,
                             sender=['192.168.12.1', 5001], packet=packet,
                             payload_base64=base64.b64encode(json.dumps(packet).encode()).decode()))
        rows.append(dict(type='session_end', complete=True))
        self.rows = rows
        self.bag.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')

    def tearDown(self):
        for observer in self.observers:
            observer.close()
        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
            self.process.communicate(timeout=5)
        self.env.stop()
        self.temp.cleanup()

    def start(self, *extra):
        self.process = subprocess.Popen([sys.executable, '-u', '-m', 'tools.vsss_bag',
                                         str(self.bag), *extra], cwd=ROOT,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + 4
        while (not self.token_path.exists() or not self.token_path.read_text()) and time.monotonic() < deadline:
            self.assertIsNone(self.process.poll())
            time.sleep(.01)
        self.assertTrue(self.token_path.exists())

    def observe(self):
        observer = ProviderClient(('udp',))
        self.observers.append(observer)
        return observer

    def test_waits_for_observer_and_fans_out_same_events_without_hardware(self):
        self.start()
        self.assertIn('Bag provider:', self.process.stdout.readline())
        self.assertIn('playback has not started', self.process.stdout.readline())
        time.sleep(1.)  # Longer than the entire bag; no frames may be consumed.
        self.assertIsNone(self.process.poll())
        first = self.observe()
        initial = first.read(2)
        self.assertIn('first feed subscriber connected', self.process.stdout.readline())
        self.assertEqual(initial['sequence'], 0)
        self.assertGreater(initial['monotonic_ns'], 1_000_000_000)
        self.assertEqual(initial['payload_base64'], self.rows[1]['payload_base64'])
        second = self.observe()
        self.assertEqual(first.read(2), second.read(2))
        self.assertEqual(first.read(2), second.read(2))
        self.assertEqual(first.read(2)['type'], 'stream_end')
        self.assertEqual(second.read(2)['type'], 'stream_end')
        self.assertTrue(first.finished)
        self.assertEqual(first.snapshot['vision']['metadata']['status']['setup']['name'], 'test-field')
        self.assertIsNone(self.process.poll())
        late = self.observe()
        self.assertEqual(late.read(2)['type'], 'stream_end')
        self.assertIsNone(self.process.poll())

    def test_immediate_plays_to_end_without_subscribers(self):
        self.start('--immediate', '--speed', '4')
        self.assertIn('Bag provider:', self.process.stdout.readline())
        self.assertIn('Playback started: --immediate', self.process.stdout.readline())
        time.sleep(.6)
        observer = self.observe()
        self.assertTrue(observer.snapshot['vision']['replay']['finished'])
        self.assertEqual(observer.read(2)['type'], 'stream_end')
        self.assertIsNone(self.process.poll())

    def test_replay_rejects_car_control_and_gps_uses_current_delivery_time(self):
        self.start()
        self.assertEqual(available_cars(), [])
        connection = ProviderConnection('car1')
        response = json.loads(connection.stdout.readline())
        self.assertFalse(response['ready'])
        self.assertIn('read-only', response['error'])
        connection.stdin.close()
        connection.stdout.close()
        connection.mark_done()
        connection.wait(1)
        gps = GPS()
        try:
            deadline = time.monotonic() + .3
            while gps.get_car(1) is None and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertIsNotNone(gps.get_car(1))
            self.assertEqual(gps.balls[0].x, .2)
        finally:
            gps.close()

    def test_separate_plot_process_uses_provider_metadata(self):
        self.start('--speed', '3')
        image = self.directory / 'plot.png'
        result = subprocess.run([sys.executable, '-m', 'tools.vsss_plot', '--width', '320',
                                 '--output', str(image)], cwd=ROOT, capture_output=True,
                                text=True, timeout=8)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(image.is_file())
        self.assertIsNone(self.process.poll())

    def test_explicit_live_stays_running_without_clients_and_refuses_competitor(self):
        self.process = subprocess.Popen([sys.executable, '-u', '-m', 'tools.vsss_live'],
                                        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + 8
        while (not self.token_path.exists() or not self.token_path.read_text()) and time.monotonic() < deadline:
            self.assertIsNone(self.process.poll())
            time.sleep(.01)
        time.sleep(5.5)
        self.assertIsNone(self.process.poll())
        result = subprocess.run([sys.executable, '-m', 'tools.vsss_bag', str(self.bag)],
                                cwd=ROOT, capture_output=True, text=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('already be active', result.stderr)

    def test_conflicting_provider_cannot_replace_active_session(self):
        self.start()
        token = self.token_path.read_text()
        result = subprocess.run([sys.executable, '-m', 'tools.vsss_bag', str(self.bag)],
                                cwd=ROOT, capture_output=True, text=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('already be active', result.stderr)
        self.assertEqual(self.token_path.read_text(), token)

    def test_loop_restarts_with_a_new_timing_cycle(self):
        self.start('--loop', '--speed', '4')
        observer = self.observe()
        events = [observer.read(2) for _ in range(4)]
        self.assertEqual([event['sequence'] for event in events], [0, 1, 2, 0])
        self.assertNotEqual(events[0]['replay']['cycle_source_id'],
                            events[-1]['replay']['cycle_source_id'])
        self.assertIsNone(self.process.poll())

    def test_corrupt_tail_is_reported_after_delivering_valid_events(self):
        with self.bag.open('a') as output:
            output.write('\n{corrupt json')
        self.start('--speed', '4')
        observer = self.observe()
        events = [observer.read(2) for _ in range(4)]
        self.assertEqual([event['sequence'] for event in events[:3]], [0, 1, 2])
        self.assertEqual(events[-1]['type'], 'stream_end')
        self.assertFalse(events[-1]['complete'])
        self.assertIn('invalid JSON', events[-1]['error'])


if __name__ == '__main__':
    unittest.main()
