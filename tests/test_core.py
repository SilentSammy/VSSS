import asyncio
from dataclasses import replace
import json
from pathlib import Path
from queue import Queue
import socket
import struct
import subprocess
import sys
import tempfile
from threading import Event as Flag, Thread
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import Pyro5.api

from vsss import *
from vsss.models import encode, decode
from vsss.live import GPSLive, CarCmdLive, VELOCITY_UUID
from vsss.playback import VSSSPlayback
from vsss.recording import Recorder
from vsss.simulation import VSSSSim


def wait_until(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("condition did not become true")


class CoreTest(unittest.TestCase):
    def test_models_and_atomic_simulation(self):
        original = WorldState(cars=[CarState(1)], balls=[BallState(0)])
        self.assertEqual(decode(json.loads(json.dumps(encode(original)))), original)
        with self.assertRaises(ValueError):
            Velocity(float('nan'))
        with VSSSSim(original) as sim:
            sim.cars.set_velocity(1, Velocity(.5))
            sim.step(.1)
            self.assertAlmostEqual(sim.gps.state().cars[0].pose.x, .03)
            before = sim.gps.state()
            with self.assertRaises(ValueError):
                sim.update_state(StatePatch(cars=[CarState(1, Pose(5, 0))]))
            self.assertEqual(sim.gps.state(), before)
            self.assertIn(1, sim.cars._owners)
            sim.update_state(StatePatch(cars=[CarState(1, Pose(.2, 0))], balls=[BallState(0, Pose(.1, .2))]))
            sim.step(.1)
            self.assertEqual(sim.gps.state().cars[0].pose.x, .2)
            self.assertNotIn(1, sim.cars._owners)

    def test_bounded_events(self):
        hub = EventHub()
        subscription = hub.subscribe('a', ['gps'], 1)
        hub.publish('gps', WorldState())
        hub.publish('gps', WorldState())
        with self.assertRaises(OverflowError):
            hub.poll('a', subscription, 0)
        hub.disconnect('a')
        self.assertEqual(hub.subscriptions, {})

    def test_rpc_callbacks_ownership_and_exposure(self):
        with tempfile.TemporaryDirectory() as directory, VSSSSim() as sim:
            endpoint = Path(directory) / 'rpc.json'
            with sim.serve(endpoint=endpoint, port=0):
                with VSSSProvider.connect(endpoint) as owner, VSSSProvider.connect(endpoint) as reader:
                    self.assertIsInstance(reader.gps.state(), WorldState)
                    events = Queue()
                    # Callback-to-command invocation crosses Pyro thread ownership.
                    def changed(state):
                        reader.cars.stop(1)
                        events.put(state)
                    reader.gps.on_change = changed
                    sim.step(.01)
                    self.assertIsInstance(events.get(timeout=3), WorldState)
                    reader.gps.on_change = None
                    owner.cars.set_velocity(1, Velocity(.2))
                    reader.gps.state()  # Reading does not take ownership.
                    self.assertEqual(sim.cars._owners[1], owner._session)
                    with self.assertRaises(ValueError):
                        reader._invoke('gps', '_update', [WorldState()], {})
                    settings = json.loads(endpoint.read_text())
                    with Pyro5.api.Proxy(settings['uri']) as bad:
                        bad._pyroHandshake = 'wrong'
                        with self.assertRaises(Exception):
                            bad.open()
                wait_until(lambda: not sim.cars._owners)
                self.assertEqual(sim._velocities[1], Velocity())
            self.assertFalse(endpoint.exists())

    def test_record_playback_and_start_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            endpoint, bag = directory / 'rpc.json', directory / 'domain.jsonl'
            with VSSSSim() as sim:
                with sim.serve(endpoint=endpoint, port=0):
                    with VSSSProvider.connect(endpoint) as client:
                        with Recorder(client, bag):
                            client.cars.set_velocity(1, Velocity(.3))
                            sim.step(.1)
                            expected = sim.gps.state()
                            # Wait for recorder to drain the known last event.
                            wait_until(lambda: '"sequence": 1' in bag.read_text())
            with VSSSPlayback(bag, speed=10) as replay:
                replay.start()
                self.assertFalse(replay.finished.wait(.05))
                with replay.serve(endpoint=endpoint, port=0):
                    with VSSSProvider.connect(endpoint) as client:
                        values = Queue()
                        client.gps.on_change = values.put
                        self.assertTrue(replay.finished.wait(3))
                        self.assertIsNone(replay.error)
                        self.assertEqual(client.gps.state(), expected)
                        self.assertTrue(client.cars.observed())
                        with self.assertRaises(RuntimeError):
                            client.cars.set_velocity(1, Velocity(.1))
                        self.assertIsInstance(values.get(timeout=3), WorldState)

    def test_subprocess_and_crash_disconnect(self):
        code = """
import sys
from vsss import VSSSProvider, Velocity
with VSSSProvider.connect(sys.argv[1]) as client:
    assert client.describe()['mode'] == 'simulation'
    client.cars.set_velocity(1, Velocity(.2))
    print('ready', flush=True)
    sys.stdin.readline()
"""
        with tempfile.TemporaryDirectory() as directory, VSSSSim() as sim:
            endpoint = Path(directory) / 'rpc.json'
            with sim.serve(endpoint=endpoint, port=0):
                process = subprocess.Popen([sys.executable, '-u', '-c', code, str(endpoint)],
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                lines = Queue()
                Thread(target=lambda: [lines.put(line.strip()) for line in process.stdout], daemon=True).start()
                try:
                    self.assertEqual(lines.get(timeout=5), 'ready')
                    self.assertIn(1, sim.cars._owners)
                    process.kill()
                    process.wait(timeout=5)
                    wait_until(lambda: not sim.cars._owners)
                    self.assertEqual(sim._velocities[1], Velocity())
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.communicate(timeout=5)

    def test_live_udp_adapter(self):
        hub = EventHub()
        gps = GPSLive(hub, server=None)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                sender.sendto(json.dumps({'board_detected': True, 'markers': [{'id': 1, 'x': .2, 'y': .3, 'angle': 45}],
                                          'balls': [{'x': .1, 'y': .1}], 'frame_sequence': 9}).encode(),
                              ('127.0.0.1', gps.listen_port))
            wait_until(lambda: gps.state().sequence == 9)
            self.assertEqual(gps.state().cars[0], CarState(1, Pose(.2, .3, 45)))
            self.assertIsNotNone(gps.age())
        finally:
            gps.close()

    def test_ble_adapter_with_fake_transport(self):
        writes = []
        class Scanner:
            @staticmethod
            async def find_device_by_filter(predicate, timeout):
                return SimpleNamespace(name='car1')
            @staticmethod
            async def discover(**kwargs):
                return {'car1': (SimpleNamespace(name='car1'), SimpleNamespace(local_name='car1'))}
        class Client:
            def __init__(self, device, timeout): self.is_connected = False
            async def connect(self): self.is_connected = True
            async def disconnect(self): self.is_connected = False
            async def write_gatt_char(self, uuid, value, response): writes.append((uuid, value))
            async def read_gatt_char(self, uuid): return struct.pack('<BBBBfffI', 0, 3, 0, 0, .1, .2, 0, 42)
        with patch.dict(sys.modules, {'bleak': SimpleNamespace(BleakClient=Client, BleakScanner=Scanner)}):
            cars = CarCmdLive(EventHub())
            try:
                self.assertEqual(cars.available(), [1])
                cars.set_velocity(1, Velocity(.5))
                self.assertEqual(writes[-1], (VELOCITY_UUID, bytes([64, 0, 0])))
                self.assertEqual(cars.status(1)['packet_count'], 42)
                before = len(writes)
                wait_until(lambda: len(writes) > before)
                cars.disconnect('local')
                self.assertEqual(writes[-1][1], bytes(3))
            finally:
                cars.close()
            self.assertEqual(writes[-1][1], bytes(3))


if __name__ == '__main__':
    unittest.main()
