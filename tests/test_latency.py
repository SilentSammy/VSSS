import asyncio
from threading import Event
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from vsss import EventHub, Velocity
from vsss.live import CarCmdLive
from vsss.teleop import LatestCommands


class LatencyTest(unittest.TestCase):
    def test_pending_inputs_coalesce_to_latest_while_write_is_blocked(self):
        entered, release, stopped = Event(), Event(), Event()
        values = []
        class Cars:
            def set_velocity(self, car_id, value):
                values.append((car_id, value))
                if len(values) == 1:
                    entered.set()
                    release.wait(3)
                else:
                    stopped.set()
        writer = LatestCommands(Cars())
        try:
            writer.submit(1, Velocity(.1))
            self.assertTrue(entered.wait(2))
            writer.submit(1, Velocity(.2))
            writer.submit(1, Velocity(-.3))
            writer.submit(1, Velocity())
            self.assertEqual(values, [(1, Velocity(.1))])
            release.set()
            self.assertTrue(stopped.wait(2))
        finally:
            release.set()
            writer.close()
        self.assertEqual(values, [(1, Velocity(.1)), (1, Velocity())])

    def test_sender_failure_is_visible_and_pending_commands_are_discarded(self):
        class Cars:
            def set_velocity(self, *args): raise ConnectionError('lost')
        writer = LatestCommands(Cars())
        writer.submit(1, Velocity())
        writer._thread.join(2)
        self.assertIsInstance(writer.error, ConnectionError)
        with self.assertRaises(RuntimeError): writer.submit(1, Velocity(.1))
        writer.close()

    def test_heartbeat_does_not_duplicate_active_command_stream(self):
        writes = []
        class Scanner:
            @staticmethod
            async def find_device_by_filter(*args, **kwargs): return object()
        class Client:
            def __init__(self, *args, **kwargs): self.is_connected = False
            async def connect(self): self.is_connected = True
            async def disconnect(self): self.is_connected = False
            async def write_gatt_char(self, uuid, value, response):
                await asyncio.sleep(.025)
                writes.append(bytes(value))
        with patch.dict('sys.modules', {'bleak': SimpleNamespace(BleakClient=Client, BleakScanner=Scanner)}):
            cars = CarCmdLive(EventHub())
            try:
                for i in range(14): cars.set_velocity(1, Velocity(.1 if i % 2 else .2))
                # Initial stop plus exactly 14 changed commands; no extra writes.
                self.assertEqual(len(writes), 15)
                before = len(writes)
                deadline = time.monotonic() + 2
                while len(writes) == before and time.monotonic() < deadline: time.sleep(.01)
                self.assertGreater(len(writes), before)  # Idle lease still renewed.
            finally:
                cars.close()

    def test_fresh_command_cancels_heartbeat_already_waiting_for_lock(self):
        class Client:
            async def write_gatt_char(self, *args, **kwargs):
                raise AssertionError('redundant heartbeat')
        cars = CarCmdLive(EventHub())
        async def run():
            lock = cars._io_lock(1)
            await lock.acquire()
            cars._clients[1] = Client()
            cars._active[1] = bytes(3)
            cars._last_write[1] = time.monotonic() - 1
            task = asyncio.create_task(cars._heartbeat_car(1))
            await asyncio.sleep(0)
            cars._last_write[1] = time.monotonic()
            lock.release()
            await task
            cars._clients.clear()
            cars._active.clear()
        try:
            cars._call(run())
        finally:
            cars.close()

    def test_discovery_does_not_block_command_lock(self):
        entered, release = Event(), Event()
        class Scanner:
            @staticmethod
            async def discover(**kwargs):
                entered.set()
                while not release.is_set(): await asyncio.sleep(.01)
                return {}
        from threading import Thread
        with patch.dict('sys.modules', {'bleak': SimpleNamespace(BleakScanner=Scanner)}):
            cars = CarCmdLive(EventHub())
            thread = Thread(target=lambda: cars.available(refresh=True))
            thread.start()
            try:
                self.assertTrue(entered.wait(2))
                self.assertTrue(cars._lock.acquire(timeout=.1))
                cars._lock.release()
            finally:
                release.set()
                thread.join(2)
                cars.close()
