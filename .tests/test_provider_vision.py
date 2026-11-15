"""Shared UDP fanout, original-byte recording, isolation and bounded observers."""
import base64
import json
import os
from queue import Queue
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from vsss._provider import Provider
from vsss.provider import ProviderClient
from vsss.gps import GPS


class VisionProviderTests(unittest.TestCase):
    def setUp(self):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        self.env = patch.dict(os.environ, {'VSSS_PROVIDER_PORT': str(port)})
        self.env.start()
        self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server.bind(('127.0.0.1', 0))
        self.server.settimeout(3)
        self.config = dict(server='127.0.0.1', subscription_port=self.server.getsockname()[1])
        self.observers = []

    def tearDown(self):
        for observer in self.observers:
            observer.close()
        self.server.close()
        self.env.stop()

    def observer(self, config=None, topics=('udp',)):
        observer = ProviderClient(topics, self.config if config is None else config)
        self.observers.append(observer)
        return observer

    def test_one_subscription_identical_datagrams_and_timestamps_for_two_processes(self):
        first = self.observer()
        code = ("import json; from vsss.provider import ProviderClient; "
                "o=ProviderClient(('udp',),json.loads(input())); "
                "print(json.dumps(o.snapshot),flush=True); print(json.dumps(o.read(4)),flush=True); o.close()")
        process = subprocess.Popen([sys.executable, '-u', '-c', code], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            process.stdin.write(json.dumps(self.config) + '\n')
            process.stdin.flush()
            snapshot = json.loads(process.stdout.readline())
            self.assertEqual(snapshot['vision'], first.snapshot['vision'])
            data, target = self.server.recvfrom(100)
            self.assertEqual(data, b'POSE_SUBSCRIBE')
            raw = b'{ "board_detected":true,"markers":[],"balls":[{"x":0.2,"y":0.3}] }'
            self.server.sendto(raw, target)
            event = first.read(3)
            other = json.loads(process.stdout.readline())
            self.assertEqual(event, other)
            self.assertEqual(base64.b64decode(event['payload_base64']), raw)
            self.assertTrue(event['valid'])
            self.assertEqual(target[1], first.snapshot['vision']['local_port'])
            # Only one sender port ever subscribes for this source.
            _, renewed = self.server.recvfrom(100)
            self.assertEqual(target, renewed)
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)

    def test_malformed_packets_preserved_and_topics_isolated(self):
        observer = self.observer()
        ble_only = self.observer(topics=('ble',))
        _, target = self.server.recvfrom(100)
        for raw in (b'\xffnot-json', b'[]', b'{"board_detected":true,"markers":[{}]}',
                    b'{"board_detected":true,"markers":{}}'):
            self.server.sendto(raw, target)
            event = observer.read(3)
            self.assertFalse(event['valid'])
            self.assertEqual(base64.b64decode(event['payload_base64']), raw)
        self.assertIsNone(ble_only.read(.1))

    def test_source_stays_subscribed_until_last_consumer_leaves(self):
        first, second = self.observer(), self.observer()
        _, target = self.server.recvfrom(100)
        first.close()
        data, address = self.server.recvfrom(100)
        self.assertEqual(data, b'POSE_SUBSCRIBE')
        self.assertEqual(address, target)
        second.close()
        while True:
            data, address = self.server.recvfrom(100)
            if data == b'POSE_UNSUBSCRIBE':
                self.assertEqual(address, target)
                break

    def test_slow_observer_reports_overflow_without_blocking_other_consumers(self):
        provider = Provider()
        slow, fast = Queue(maxsize=1), Queue(maxsize=4)
        topics = frozenset({'udp'})
        provider.observers.update({(slow, topics, 'source'), (fast, topics, 'source')})
        event = dict(type='udp', sequence=0)
        provider.publish(event, 'source')
        provider.publish(dict(event, sequence=1), 'source')
        self.assertEqual(slow.get_nowait()['type'], 'observer_overflow')
        self.assertEqual([fast.get_nowait()['sequence'], fast.get_nowait()['sequence']], [0, 1])

    def test_gps_recovers_after_provider_failure_and_observers_detect_disconnect(self):
        # Own the exact test process so no unrelated provider can be terminated.
        process = subprocess.Popen([sys.executable, '-m', 'vsss._provider'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        gps = None
        try:
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                try:
                    with socket.create_connection(('127.0.0.1', int(os.environ['VSSS_PROVIDER_PORT'])), timeout=.1):
                        break
                except OSError:
                    time.sleep(.02)
            gps = GPS(port=0, subscribe=False)
            observer = self.observer(dict(server=None, port=0, listen_port=0))
            packet = json.dumps(dict(board_detected=True, markers=[dict(id=1, x=.1, y=.2, angle=90)])).encode()
            target = ('127.0.0.1', gps.port)
            self.server.sendto(packet, target)
            self.assertTrue(observer.read(2)['valid'])
            deadline = time.monotonic() + 2
            while gps.get_car(1) is None and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertIsNotNone(gps.get_car(1))
            old_source = observer.snapshot['vision']['source_id']
            process.kill()
            process.wait(3)
            with self.assertRaises((ConnectionError, OSError)):
                observer.read(3)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if gps._observer.snapshot['vision']['source_id'] != old_source:
                    break
                time.sleep(.02)
            self.assertNotEqual(gps._observer.snapshot['vision']['source_id'], old_source)
            self.assertIsNone(gps.get_car(1))
            self.server.sendto(packet, ('127.0.0.1', gps.port))
            deadline = time.monotonic() + 2
            while gps.get_car(1) is None and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertIsNotNone(gps.get_car(1))
        finally:
            if gps:
                gps.close()
            if process.poll() is None:
                process.kill()
                process.wait(3)

    def test_recorder_marks_stream_failure_incomplete(self):
        # A synthetic observer failure verifies the recorder's on-disk contract.
        from tools.vsss_observer import main
        from unittest.mock import Mock
        observer = Mock()
        observer.__enter__ = Mock(return_value=observer)
        observer.__exit__ = Mock(return_value=False)
        observer.snapshot = dict(type='provider_snapshot', vision=None, controlled_cars=[])
        observer.read.side_effect = ConnectionError('provider interrupted')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'interrupted.jsonl'
            with patch('tools.vsss_observer.ProviderClient', return_value=observer):
                with self.assertRaises(ConnectionError):
                    main([str(output), '--topics', 'ble'])
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(rows[-1]['type'], 'session_end')
            self.assertFalse(rows[-1]['complete'])


if __name__ == '__main__':
    unittest.main()
