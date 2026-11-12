"""Integration checks for shared BLE ownership and passive recording."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from vsss._ble_client import MecanumBLEClient
from vsss._provider_connection import connect

ROOT = Path(__file__).resolve().parents[1]


class ProviderIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.log = Path(self.temp.name) / 'ble.jsonl'
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        self.env = patch.dict(os.environ, {'VSSS_PROVIDER_PORT': str(port),
            'PYTHONPATH': str(ROOT / '.tests/fake_ble') + os.pathsep + str(ROOT),
            'BLE_TEST_LOG': str(self.log), 'PYTHONDONTWRITEBYTECODE': '1'})
        self.env.start()
        self.cars = []

    def tearDown(self):
        for car in self.cars:
            car.disconnect()
        self.env.stop()
        self.temp.cleanup()

    def car(self, name):
        car = MecanumBLEClient(name)
        self.cars.append(car)
        car.connect()
        car.wait_until_connected(6)
        return car

    def test_observer_sees_actual_writes_and_multiple_cars_are_exclusive(self):
        observer = connect('observe')
        observer.settimeout(8)
        stream = observer.makefile('r', encoding='utf-8')
        self.assertEqual(json.loads(stream.readline()), {'provider': 'observing'})
        a, b = self.car('car1'), self.car('car2')
        duplicate = MecanumBLEClient('car1')
        try:
            duplicate.connect()
            with self.assertRaises(TimeoutError):
                duplicate.wait_until_connected(3)
            self.assertIn('already controlled', duplicate.last_error)
        finally:
            duplicate.disconnect()
        a.set_velocity({'x': .2})
        time.sleep(.25)
        a.stop()
        b.set_velocity({'y': .2})
        time.sleep(.25)
        b.stop()
        seen = set()
        end = time.monotonic() + 5
        while time.monotonic() < end and seen != {'car1', 'car2'}:
            event = json.loads(stream.readline())
            if (event.get('type') == 'ble' and event.get('phase') == 'start' and
                    event.get('operation') == 'write' and event.get('uuid', '').endswith('def1') and
                    event.get('payload_hex') not in ('000000', None)):
                seen.add(event['car'])
                self.assertIn('session_id', event)
                self.assertIn('connection_id', event)
        self.assertEqual(seen, {'car1', 'car2'})
        stream.close()
        observer.close()

    def test_controller_process_exit_stops_while_observer_remains(self):
        observer = connect('observe')
        observer.settimeout(8)
        stream = observer.makefile('r', encoding='utf-8')
        self.assertEqual(json.loads(stream.readline()), {'provider': 'observing'})
        code = ("from vsss._ble_client import MecanumBLEClient; import os,time; "
                "c=MecanumBLEClient('car1'); c.connect(); c.wait_until_connected(6); "
                "c.set_velocity({'x':.4}); time.sleep(.5); os._exit(0)")
        subprocess.run([sys.executable, '-c', code], check=True, timeout=12)
        deadline = time.monotonic() + 6
        events = []
        while time.monotonic() < deadline:
            events = [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []
            writes = [e['data'] for e in events if e['event'] == 'write']
            if any(any(x) for x in writes) and writes[-1] == [0, 0, 0]:
                break
            time.sleep(.05)
        self.assertTrue(any(any(x) for x in writes), events)
        self.assertEqual(writes[-1], [0, 0, 0])
        stream.close()
        observer.close()

    def test_recorder_combines_udp_and_ble(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server.bind(('127.0.0.1', 0))
        server.settimeout(5)
        packet = {'board_detected': True, 'markers': [], 'balls': [{'x': .2, 'y': .4}]}
        def serve():
            try:
                data, address = server.recvfrom(100)
                if data == b'POSE_SUBSCRIBE':
                    server.sendto(json.dumps(packet).encode(), address)
            except OSError:
                pass
        sender = threading.Thread(target=serve, daemon=True)
        sender.start()
        path = Path(self.temp.name) / 'session.jsonl'
        process = subprocess.Popen([sys.executable, '-u', '-m', 'tools.vsss_observer',
                                    '--server', '127.0.0.1', '--subscription-port', str(server.getsockname()[1]),
                                    '--duration', '4', '--output', str(path)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 5
            while (not path.exists() or path.stat().st_size == 0) and time.monotonic() < deadline:
                time.sleep(.02)
            car = self.car('car1')
            car.set_velocity({'x': .2})
            time.sleep(.3)
            car.stop()
            stdout, stderr = process.communicate(timeout=8)
            self.assertEqual(process.returncode, 0, stderr)
            events = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertTrue(any(e.get('type') == 'udp' and e.get('packet') == packet for e in events))
            self.assertTrue(any(e.get('type') == 'ble' and e.get('phase') == 'complete' and
                                e.get('operation') == 'write' and e.get('car') == 'car1' for e in events))
            self.assertTrue(events[-1]['complete'])
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
            server.close()
            sender.join(6)

    def test_concurrent_first_clients_share_provider(self):
        barrier = threading.Barrier(2)
        errors = []
        clients = []
        def start(name):
            try:
                client = MecanumBLEClient(name)
                clients.append(client)
                barrier.wait(timeout=5)
                client.connect()
                client.wait_until_connected(10)
            except Exception as error:
                errors.append(error)
        threads = [threading.Thread(target=start, args=(name,)) for name in ('car1', 'car2')]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(15)
            self.assertFalse(errors, errors)
            self.assertTrue(all(client.is_connected for client in clients))
            self.assertEqual(len(clients), 2)
        finally:
            for client in clients:
                client.disconnect()

    def test_provider_restarts_after_idle_shutdown(self):
        self.car('car1').disconnect()
        time.sleep(6)
        self.car('car1').stop()


if __name__ == '__main__':
    unittest.main()
