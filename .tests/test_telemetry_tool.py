import contextlib
import io
import json
from pathlib import Path
import socket
import tempfile
from threading import Thread
import unittest
from unittest.mock import patch

from tools.telemetry_listener import main


class TelemetryToolTests(unittest.TestCase):
    def test_unicast_recording_and_unsubscribe(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server.bind(('127.0.0.1', 0))
        server.settimeout(3)
        messages = []
        packet = {'board_detected': True, 'markers': [{'id': 1, 'x': .1, 'y': .2, 'angle': 90}]}

        def serve():
            try:
                while True:
                    data, client = server.recvfrom(1024)
                    messages.append(data)
                    if data == b'POSE_SUBSCRIBE':
                        server.sendto(json.dumps(packet).encode(), client)
                    elif data == b'POSE_UNSUBSCRIBE':
                        break
            except OSError:
                pass

        worker = Thread(target=serve, daemon=True)
        worker.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'packets.jsonl'
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    main(['--server', '127.0.0.1', '--subscription-port', str(server.getsockname()[1]),
                          '--duration', '.4', '--output', str(path), '--quiet'])
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(rows[0]['packet'], packet)
                self.assertEqual(rows[0]['sender'], '127.0.0.1')
                self.assertIn('1 valid packet(s)', output.getvalue())
                self.assertNotIn('ID 1:', output.getvalue())
            worker.join(2)
            self.assertEqual(messages, [b'POSE_SUBSCRIBE', b'POSE_UNSUBSCRIBE'])
        finally:
            server.close()
            worker.join(3)

    def test_invalid_options_fail_before_opening_socket(self):
        for args in (['--port', '0'], ['--server', '127.0.0.1', '--port', '65535'],
                     ['--listen-port', '5'], ['--duration', 'nan']):
            with self.subTest(args=args), patch('tools.telemetry_listener.socket.socket') as sock, \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main(args)
                sock.assert_not_called()


if __name__ == '__main__':
    unittest.main()
