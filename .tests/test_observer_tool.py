import contextlib
import io
import json
from pathlib import Path
import socket
import tempfile
from threading import Thread
import unittest
from unittest.mock import MagicMock, patch

from tools.vsss_observer import main, format_event


class ObserverToolTests(unittest.TestCase):
    def test_live_view_needs_no_recording_and_preserves_ball_and_ble_details(self):
        observer = MagicMock()
        observer.__enter__.return_value = observer
        observer.snapshot = dict(type='provider_snapshot', controlled_cars=['car1'], vision=None)
        udp = dict(type='udp', valid=True, sender=['127.0.0.1', 5001],
                   packet=dict(board_detected=True, markers=[], balls=[dict(x=.3, y=.4)]))
        ble = dict(type='ble', car='car1', operation='write', characteristic='velocity',
                   request=7, phase='start', reason='command', value=dict(x=.2, y=0, w=0))
        observer.read.side_effect = [udp, ble, KeyboardInterrupt()]
        with patch('tools.vsss_observer.ProviderObserver', return_value=observer), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            main(['--topics', 'udp', 'ble'])
        text = output.getvalue()
        self.assertIn('ball: x=+0.300m, y=+0.400m', text)
        self.assertIn('BLE car1 write velocity #7 start (command)', text)
        self.assertIn('"x": 0.2', text)
        self.assertNotIn('Recording to', text)

    def test_invalid_packets_and_ble_errors_remain_readable(self):
        self.assertIn('invalid: bad json', format_event(dict(type='udp', valid=False,
            sender=['127.0.0.1', 5001], error='bad json')))
        self.assertIn('timeout', format_event(dict(type='ble', car='car1', operation='write',
            characteristic='velocity', phase='error', request=8, error='timeout')))

    def test_ble_status_is_decoded_without_changing_recorded_event(self):
        from vsss._ble_protocol import STATUS_STRUCT
        event = dict(type='ble', car='car1', operation='read', characteristic='status',
                     phase='complete', request=9,
                     payload_hex=STATUS_STRUCT.pack(0, 1, 0, 0, .1, .2, 90, 123).hex())
        original = event.copy()
        text = format_event(event)
        self.assertIn('"mode": "idle"', text)
        self.assertIn('"packet_count": 123', text)
        self.assertEqual(event, original)

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
                event = next(row for row in rows if row['type'] == 'udp')
                self.assertEqual(event['packet'], packet)
                self.assertEqual(event['sender'][0], '127.0.0.1')
                self.assertTrue(rows[-1]['complete'])
                self.assertIn('1 UDP', output.getvalue())
                self.assertNotIn('ID 1:', output.getvalue())
            worker.join(2)
            self.assertEqual(messages, [b'POSE_SUBSCRIBE', b'POSE_UNSUBSCRIBE'])
        finally:
            server.close()
            worker.join(3)

    def test_invalid_options_fail_before_opening_socket(self):
        for args in (['--port', '0'], ['--server', '127.0.0.1', '--port', '65535'],
                     ['--quiet'], ['--duration', 'nan']):
            with self.subTest(args=args), patch('tools.vsss_observer.ProviderObserver') as sock, \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main(args)
                sock.assert_not_called()


if __name__ == '__main__':
    unittest.main()
