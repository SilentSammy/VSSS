import contextlib
import io
from threading import Event
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from vsss import EventHub, Velocity
from vsss.live import CarCmdLive
from vsss.teleop import LatestCommands
from vsss import VSSSProvider
from vsss.simulation import VSSSSim


def wait_for(predicate):
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        if predicate(): return
        time.sleep(.01)
    raise AssertionError('timed out')


class ReconnectTest(unittest.TestCase):
    def test_reconnecting_sender_uses_rpc_and_releases_its_own_motion(self):
        with tempfile.TemporaryDirectory() as directory, VSSSSim() as sim:
            endpoint = Path(directory) / 'rpc.json'
            with sim.serve(endpoint=endpoint, port=0), VSSSProvider.connect(endpoint) as client:
                with contextlib.redirect_stdout(io.StringIO()):
                    writer = LatestCommands(client.cars, client.fork, retry_connections=True)
                    try:
                        writer.submit(1, Velocity())
                        wait_for(lambda: writer._threads[1].is_alive())
                        # connect can discard the initial zero, but remembers neutral.
                        wait_for(lambda: sim.cars.connection(1)['connected'])
                        time.sleep(.1)
                        writer.submit(1, Velocity(.2))
                        wait_for(lambda: sim._velocities.get(1) == Velocity(.2))
                    finally:
                        writer.close()
                self.assertIsNone(writer.error)
                self.assertEqual(sim._velocities[1], Velocity())

    def test_missing_car_retries_without_blocking_other_car_and_requires_neutral(self):
        class Cars:
            def __init__(self):
                self.present = False
                self.generation = 1
                self.attempts = 0
                self.commands = []
                self.released = []
            def connect(self, car):
                if car == 1:
                    self.attempts += 1
                    if not self.present: raise ConnectionError('not found')
                return self.connection(car)
            def connection(self, car):
                return dict(connected=car == 2 or self.present, generation=self.generation)
            def set_velocity(self, car, value, *, generation):
                if not self.connection(car)['connected'] or generation != self.generation:
                    raise ConnectionError('connection changed')
                self.commands.append((car, value, generation))
            def release(self, car): self.released.append(car)
        cars = Cars()
        with contextlib.redirect_stdout(io.StringIO()):
            writer = LatestCommands(cars, retry_connections=True)
            try:
                writer.submit(1, Velocity(.5))
                writer.submit(2, Velocity())
                wait_for(lambda: cars.attempts >= 2)
                writer.submit(2, Velocity(.2))
                wait_for(lambda: (2, Velocity(.2), 1) in cars.commands)
                cars.present = True
                wait_for(lambda: cars.attempts >= 3)
                # A held control from before discovery cannot start the car.
                time.sleep(.05)
                self.assertFalse(any(car == 1 for car, _, _ in cars.commands))
                writer.submit(1, Velocity())
                wait_for(lambda: (1, Velocity(), 1) in cars.commands)
                writer.submit(1, Velocity(.3))
                wait_for(lambda: (1, Velocity(.3), 1) in cars.commands)
                cars.present = False
                previous = cars.attempts
                wait_for(lambda: cars.attempts > previous)  # Idle polling detects loss.
                cars.generation = 2
                cars.present = True
                previous = cars.attempts
                wait_for(lambda: cars.attempts > previous)
                time.sleep(.05)
                self.assertFalse(any(car == 1 and gen == 2 for car, _, gen in cars.commands))
                writer.submit(1, Velocity())
                wait_for(lambda: (1, Velocity(), 2) in cars.commands)
                writer.submit(1, Velocity(.4))
                wait_for(lambda: (1, Velocity(.4), 2) in cars.commands)
                self.assertIsNone(writer.error)
            finally:
                writer.close()
        self.assertEqual(set(cars.released), {1, 2})

    def test_guarded_live_command_never_reconnects_or_replays_motion(self):
        scans, writes = [], []
        class Scanner:
            @staticmethod
            async def find_device_by_filter(*args, **kwargs):
                scans.append(1)
                return object()
        class Client:
            def __init__(self, *args, **kwargs): self.is_connected = False
            async def connect(self): self.is_connected = True
            async def disconnect(self): self.is_connected = False
            async def write_gatt_char(self, uuid, value, response): writes.append(value)
        with patch.dict('sys.modules', {'bleak': SimpleNamespace(BleakClient=Client, BleakScanner=Scanner)}):
            cars = CarCmdLive(EventHub())
            try:
                generation = cars.connect(1)['generation']
                cars.set_velocity(1, Velocity(.2), generation=generation)
                cars._clients[1].is_connected = False
                with self.assertRaises(ConnectionError):
                    cars.set_velocity(1, Velocity(.7), generation=generation)
                self.assertEqual(len(scans), 1)
                current = cars.connect(1)['generation']
                self.assertGreater(current, generation)
                self.assertEqual(writes[-1], bytes(3))
                with self.assertRaises(ConnectionError):
                    cars.set_velocity(1, Velocity(.7), generation=generation)
                self.assertEqual(writes[-1], bytes(3))
                # A heartbeat-failed link must also force fresh discovery.
                cars._active.pop(1)
                self.assertFalse(cars.connection(1)['connected'])
                cars.connect(1)
                self.assertEqual(len(scans), 3)
            finally:
                cars.close()

    def test_close_interrupts_retry_delay(self):
        attempted = Event()
        class Cars:
            def connect(self, car):
                attempted.set()
                raise ConnectionError('not found')
            def release(self, car): pass
        with contextlib.redirect_stdout(io.StringIO()):
            writer = LatestCommands(Cars(), retry_connections=True)
            writer.submit(1, Velocity())
            self.assertTrue(attempted.wait(1))
            started = time.monotonic()
            writer.close()
            self.assertLess(time.monotonic()-started, .3)
            self.assertIsNone(writer.error)
