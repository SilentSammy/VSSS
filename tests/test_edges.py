from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
from queue import Queue
import subprocess
import sys
import tempfile
from threading import Thread
import time
import unittest

from vsss import *
from vsss.live import GPSLive
from vsss.models import encode
from vsss.playback import VSSSPlayback
from vsss.recording import read_events
from vsss.rpc import RemoteError
from vsss.simulation import VSSSSim, scenario_state


class EdgesTest(unittest.TestCase):
    def test_new_format_fixture_and_loop(self):
        with VSSSPlayback('examples/recording.jsonl', speed=10, wait_for_subscriber=False) as player:
            player.start()
            self.assertTrue(player.finished.wait(2))
            self.assertIsNone(player.error)
            self.assertEqual(player.gps.state().cars[0].pose.x, .1)
        with VSSSPlayback('examples/recording.jsonl', speed=10, wait_for_subscriber=False, loop=True) as player:
            player.start()
            self.assertFalse(player.finished.wait(.1))
        self.assertTrue(player.finished.is_set())
        self.assertIsInstance(scenario_state('examples/scenario.jsonl'), WorldState)
        self.assertIsInstance(scenario_state('examples/recording.jsonl'), WorldState)

    def test_incomplete_recording_and_rejected_legacy_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.jsonl'
            path.write_text(json.dumps({'schema': 'vsss.recording/v1', 'state': encode(WorldState())}) + '\n')
            with self.assertRaises(ValueError):
                list(read_events(path))
            with VSSSPlayback(path, wait_for_subscriber=False) as player:
                player.start()
                self.assertTrue(player.finished.wait(2))
                self.assertIsInstance(player.error, ValueError)
            path.write_text('{"type":"bag_header","schema":3}\n')
            with self.assertRaisesRegex(ValueError, 'legacy'):
                VSSSPlayback(path)

    def test_metadata_http(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                payload = b'{"setup":{"board":{"dimensions_m":[2,3]}}}'
                self.send_response(200)
                self.end_headers()
                self.wfile.write(payload)
            def log_message(self, *args): pass
        http = HTTPServer(('127.0.0.1', 0), Handler)
        thread = Thread(target=http.serve_forever)
        thread.start()
        gps = GPSLive(EventHub(), server=None, metadata_url=f'http://127.0.0.1:{http.server_port}/status')
        try:
            self.assertEqual(gps.refresh_field(), Field(2, 3))
        finally:
            gps.close()
            http.shutdown()
            thread.join()
            http.server_close()

    def test_backend_exception_envelope(self):
        class CustomBackendError(Exception): pass
        class FaultySim(VSSSSim):
            @rpc_method
            def fail(self): raise CustomBackendError('synthetic backend failure')
        with tempfile.TemporaryDirectory() as directory, FaultySim() as provider:
            endpoint = Path(directory) / 'rpc.json'
            with provider.serve(endpoint=endpoint, port=0):
                with VSSSProvider.connect(endpoint) as client:
                    with self.assertRaisesRegex(RemoteError, 'CustomBackendError: synthetic backend failure'):
                        client.fail()

    def test_simulator_cli_process(self):
        with tempfile.TemporaryDirectory() as directory:
            endpoint = Path(directory) / 'rpc.json'
            process = subprocess.Popen([sys.executable, '-u', '-m', 'vsss.simulation', 'examples/scenario.jsonl',
                                        '--endpoint', str(endpoint), '--port', '0'],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output = Queue()
            Thread(target=lambda: [output.put(line.strip()) for line in process.stdout], daemon=True).start()
            try:
                self.assertIn('provider ready', output.get(timeout=5))
                with VSSSProvider.connect(endpoint) as client:
                    self.assertEqual(client.gps.state().cars[0].pose.x, -.2)
                    result = client.update_state(StatePatch(balls=[BallState(0, Pose(.1, .1))]))
                    self.assertEqual(result.balls[0].pose.x, .1)
            finally:
                process.terminate()
                process.communicate(timeout=5)


if __name__ == '__main__':
    unittest.main()
