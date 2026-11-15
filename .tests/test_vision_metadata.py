import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import socket
import tempfile
from threading import Thread
import time
import unittest
from unittest.mock import patch

from tools.vsss_observer import main
from vsss._provider_vision import VisionSource, source_settings
from vsss.provider import BagProvider


class VisionMetadataTests(unittest.TestCase):
    def test_polling_caches_health_config_and_preserves_last_success_on_failure(self):
        class Handler(BaseHTTPRequestHandler):
            requests = 0
            def log_message(self, *args): pass
            def do_GET(self):
                Handler.requests += 1
                payload = json.dumps(dict(schema='aruco-gps.status/v1', healthy=False,
                                          setup=dict(name='lab'))).encode()
                self.send_response(503)
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        Thread(target=server.serve_forever, daemon=True).start()
        with patch('vsss._provider_vision.METADATA_POLL_SECONDS', .05):
            source = VisionSource(source_settings(dict(server=None, port=0,
                metadata_url=f'http://127.0.0.1:{server.server_port}/api/v1/status')), lambda event: None)
            try:
                self.assertTrue(source.metadata_ready.wait(2))
                self.assertEqual(source.snapshot()['metadata']['status']['setup']['name'], 'lab')
                deadline = time.monotonic() + 2
                while Handler.requests < 2 and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertGreaterEqual(Handler.requests, 2)
                server.shutdown()
                server.server_close()
                time.sleep(.15)
                self.assertIsNotNone(source.metadata['error'])
                self.assertEqual(source.metadata['status']['setup']['name'], 'lab')
            finally:
                source.close()
                server.server_close()

    def test_bag_header_is_first_and_replayed_metadata_is_preserved(self):
        class Observer:
            snapshot = dict(type='provider_snapshot', controlled_cars=[], vision=dict(
                local_port=5001, metadata=dict(status=dict(setup=dict(name='lab')))))
            def __enter__(self): return self
            def __exit__(self, *exc): pass
            def read(self, timeout): raise KeyboardInterrupt
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bag.jsonl'
            with patch('tools.vsss_observer.ProviderObserver', return_value=Observer()), \
                    contextlib.redirect_stdout(io.StringIO()):
                main(['--topics', 'udp', '--output', str(path), '--quiet'])
            header = json.loads(path.read_text().splitlines()[0])
            self.assertEqual(header['type'], 'bag_header')
            self.assertEqual(header['schema'], 3)
            with BagProvider(path) as provider:
                self.assertEqual(provider.snapshot['vision']['metadata']['status']['setup']['name'], 'lab')


if __name__ == '__main__':
    unittest.main()
