"""Behavior tests for process cleanup, heartbeats, and the firmware lease.

Run with: python -m unittest discover -s tests -v
No Bluetooth adapter or motor hardware is used.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

from mecanum_client import MecanumBLEClient

ROOT = Path(__file__).resolve().parents[1]


class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.log = Path(self.temp.name) / 'ble.jsonl'
        self.environment = patch.dict(os.environ, {
            'PYTHONPATH': str(ROOT / 'tests/fake_ble') + os.pathsep + str(ROOT),
            'BLE_TEST_LOG': str(self.log), 'PYTHONDONTWRITEBYTECODE': '1',
        })
        self.environment.start()
        self.client = MecanumBLEClient('test')

    def tearDown(self):
        self.client.disconnect()
        self.environment.stop()
        self.temp.cleanup()

    def events(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def wait_for(self, predicate, timeout=4):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.01)
        self.fail('Timed out waiting for the worker')

    def test_ready_only_after_gatt_and_initial_stop(self):
        self.client.connect()
        process = self.client._process
        self.client.connect()
        self.assertIs(process, self.client._process)
        self.wait_for(lambda: any(e['event'] == 'link' for e in self.events()))
        self.assertFalse(self.client.is_connected)
        self.client.wait_until_connected(4)
        self.assertEqual([e for e in self.events() if e['event'] == 'write'][0]['data'], [0, 0, 0])

    def test_latched_velocity_has_heartbeats_and_stop_is_acknowledged(self):
        self.client.connect()
        self.client.wait_until_connected(4)
        self.client.set_velocity({'x': .5})
        self.wait_for(lambda: len([e for e in self.events() if e.get('data') == [64, 0, 0]]) >= 3)
        self.client.stop()
        self.assertEqual([e for e in self.events() if e['event'] == 'write'][-1]['data'], [0, 0, 0])
        self.client.disconnect()
        events = self.events()
        self.assertEqual(events[-1]['event'], 'disconnect')
        self.assertEqual(events[-2]['data'], [0, 0, 0])

    def test_newest_velocity_wins(self):
        self.client.connect()
        self.client.wait_until_connected(4)
        for _ in range(20):
            self.client.set_velocity({'x': .2})
            self.client.set_velocity({'x': -.2})
        self.client.set_velocity({'x': .5})
        self.wait_for(lambda: self.events()[-1].get('data') == [64, 0, 0])

    def test_close_cancels_a_long_discovery(self):
        self.client.device_name = 'missing'
        self.client.connect()
        self.wait_for(lambda: bool(self.events()))
        started = time.monotonic()
        self.client.disconnect()
        self.assertLess(time.monotonic() - started, 2)
        self.assertFalse(self.client.is_connected)

    def test_reconnect_starts_stopped(self):
        with patch.dict(os.environ, {'BLE_TEST_DROP_ON_MOVE': '1'}):
            self.client.connect()
        self.client.wait_until_connected(4)
        self.client.set_velocity({'x': .5})
        self.wait_for(lambda: sum(e['event'] == 'ready' for e in self.events()) == 2)
        self.wait_for(lambda: self.client.is_connected)
        time.sleep(.25)
        events = self.events()
        second_ready = [i for i, e in enumerate(events) if e['event'] == 'ready'][1]
        writes = [e['data'] for e in events[second_ready:] if e['event'] == 'write']
        self.assertTrue(writes)
        self.assertTrue(all(data == [0, 0, 0] for data in writes))

    def test_abrupt_parent_exit_stops_and_disconnects_worker(self):
        code = """
import os, time
from mecanum_client import MecanumBLEClient
car = MecanumBLEClient('test')
car.connect()
car.wait_until_connected(4)
car.set_velocity({'x': .5})
time.sleep(.1)
os._exit(0)
"""
        subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, timeout=8)
        self.wait_for(lambda: self.events() and self.events()[-1]['event'] == 'disconnect')
        events = self.events()
        self.assertTrue(any(e.get('data') == [64, 0, 0] for e in events))
        self.assertEqual(events[-2]['data'], [0, 0, 0])


class Clock:
    period = 1 << 30
    now = 0

    def ticks_ms(self):
        return self.now

    def sleep_ms(self, delay):
        pass

    def ticks_add(self, value, delta):
        return (value + delta) % self.period

    def ticks_diff(self, a, b):
        return (a - b + self.period // 2) % self.period - self.period // 2


class Radio:
    def __init__(self):
        self.log = []
        self.handler = None
        self.payload = bytes(3)

    def active(self, value): pass
    def config(self, **kwargs): pass
    def irq(self, handler): self.handler = handler
    def gap_advertise(self, interval, data=None): self.log.append(('advertise', interval))
    def gap_disconnect(self, handle): self.log.append(('disconnect', handle))
    def gatts_read(self, handle): return self.payload
    def gatts_write(self, handle, data): pass
    def gatts_register_services(self, services): return ((1,),)


class FirmwareTests(unittest.TestCase):
    def setUp(self):
        self.clock, self.radio = Clock(), Radio()
        bluetooth = types.SimpleNamespace(BLE=lambda: self.radio, UUID=lambda value: value,
                                          FLAG_READ=2, FLAG_WRITE=8, FLAG_NOTIFY=16)
        spec = importlib.util.spec_from_file_location('firmware_test', ROOT / 'mp/ble_server.py')
        self.server = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'bluetooth': bluetooth, 'time': self.clock}):
            spec.loader.exec_module(self.server)
        self.server.control_callbacks = {'velocity': lambda data: None}
        self.server.disconnect_callback = lambda: self.radio.log.append(('motors', 'stop'))
        self.server.start()

    def test_timeout_stops_before_disconnect_and_resumes_advertising(self):
        self.radio.handler(1, (7, 0, b''))
        self.radio.handler(3, (7, 1))
        self.clock.now = 1499
        self.server.poll()
        self.assertNotIn(('disconnect', 7), self.radio.log)
        self.clock.now = 1500
        self.server.poll()
        self.assertEqual(self.radio.log[-2:], [('motors', 'stop'), ('disconnect', 7)])
        self.radio.handler(2, (7, 0, b''))
        self.assertEqual(self.radio.log[-1], ('advertise', 20000))
        self.assertEqual(self.radio.log.count(('motors', 'stop')), 1)

    def test_initial_setup_has_longer_timeout_and_heartbeat_refreshes_it(self):
        self.radio.handler(1, (7, 0, b''))
        self.clock.now = 9000
        self.server.poll()
        self.assertNotIn(('disconnect', 7), self.radio.log)
        self.radio.handler(3, (7, 1))
        self.clock.now = 10000
        self.server.poll()
        self.assertNotIn(('disconnect', 7), self.radio.log)
        self.clock.now = 10500
        self.server.poll()
        self.assertIn(('disconnect', 7), self.radio.log)

    def test_expiry_across_clock_wrap(self):
        self.clock.now = self.clock.period - 100
        self.radio.handler(1, (7, 0, b''))
        self.radio.handler(3, (7, 1))
        self.clock.now = 1400
        self.server.poll()
        self.assertIn(('disconnect', 7), self.radio.log)

    def test_only_one_owner_and_stop_does_not_restart_advertising(self):
        self.radio.handler(1, (7, 0, b''))
        self.radio.handler(1, (8, 0, b''))
        self.assertEqual(self.server._connections, {7})
        self.assertIn(('disconnect', 8), self.radio.log)
        handler = self.radio.handler
        self.server.stop()
        count = len(self.radio.log)
        handler(2, (7, 0, b''))
        self.assertEqual(len(self.radio.log), count)


class MotorCommandTests(unittest.TestCase):
    def test_changing_speeds_fit_four_timers_and_release_all_channels(self):
        class Pin:
            OUT = 1
            def __init__(self, *args): pass
            def value(self, value): pass

        class PWM:
            live = []
            def __init__(self, pin, freq):
                frequencies = {p.frequency for p in self.live}
                if freq not in frequencies and len(frequencies) == 4:
                    raise RuntimeError('out of PWM timers')
                self.frequency = freq
                self.live.append(self)
                if len(self.live) > 6:
                    raise RuntimeError('out of PWM channels')
            def freq(self, value=None):
                if value is not None:
                    frequencies = {p.frequency for p in self.live}
                    if value not in frequencies and len(frequencies) == 4:
                        raise RuntimeError('out of PWM timers')
                    self.frequency = value
                return self.frequency
            def duty_u16(self, duty): pass
            def deinit(self): self.live.remove(self)

        spec = importlib.util.spec_from_file_location('motor_test', ROOT / 'mp/mecanum.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'machine': types.SimpleNamespace(Pin=Pin, PWM=PWM)}):
            spec.loader.exec_module(module)
        car = module.MecanumCar()
        try:
            for i in range(100):
                car.set_velocity((i % 11 - 5) / 10, ((i * 3) % 11 - 5) / 10,
                                 ((i * 7) % 11 - 5) / 10)
                self.assertLessEqual(len(PWM.live), 4)
        finally:
            car.stop()
        self.assertEqual(PWM.live, [])

    def test_three_axes_are_applied_in_one_motor_update(self):
        spec = importlib.util.spec_from_file_location('motor_test', ROOT / 'mp/mecanum.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'machine': types.SimpleNamespace(Pin=None, PWM=None)}):
            spec.loader.exec_module(module)
        motors = [types.SimpleNamespace(set_power=unittest.mock.Mock()) for _ in range(4)]
        car = module.MecanumCar(*motors)
        car.set_velocity(.2, -.1, .1)
        for motor, expected in zip(motors, (.2, .2, 0, .4)):
            motor.set_power.assert_called_once()
            self.assertAlmostEqual(motor.set_power.call_args.args[0], expected)


if __name__ == '__main__':
    unittest.main()
