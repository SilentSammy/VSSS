"""Behavior tests for process cleanup, heartbeats, and the firmware lease.

Run with: python -m unittest discover -s .tests -v
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

from vsss._ble_client import MecanumBLEClient, parse_waypoint_input

ROOT = Path(__file__).resolve().parents[1]


class WaypointConsoleTests(unittest.TestCase):
    def test_accepts_spaces_or_commas(self):
        self.assertEqual(parse_waypoint_input('0.2 0.75 -90'),
                         ('waypoint', (0.2, 0.75, -90.0)))
        self.assertEqual(parse_waypoint_input('0.2, 0.75, 90'),
                         ('waypoint', (0.2, 0.75, 90.0)))

    def test_recognizes_console_commands(self):
        for command in ('stop', 'manual', 'quit', 'help'):
            self.assertEqual(parse_waypoint_input(command), (command, None))
        self.assertEqual(parse_waypoint_input('exit'), ('quit', None))

    def test_rejects_malformed_or_nonfinite_targets(self):
        for text in ('1 2', 'a b c', 'nan 0 0', '0 inf 0'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_waypoint_input(text)


class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.log = Path(self.temp.name) / 'ble.jsonl'
        self.environment = patch.dict(os.environ, {
            'PYTHONPATH': str(ROOT / '.tests/fake_ble') + os.pathsep + str(ROOT),
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

    def test_trim_is_written_read_back_and_not_a_velocity(self):
        self.client.connect()
        self.client.wait_until_connected(4)
        self.assertEqual(self.client.trim_matrix, ((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)))
        actual = self.client.set_trim(((.9,0.,0.),(0.,1.1,0.),(0.,0.,1.)))
        self.assertAlmostEqual(actual[0][0], .9, places=6)
        self.assertAlmostEqual(actual[1][1], 1.1, places=6)
        self.assertTrue(any(e['event'] == 'trim_matrix' for e in self.events()))
        self.client.stop()
        self.assertTrue(all(e['data'] == [0, 0, 0] for e in self.events() if e['event'] == 'write'))
        for invalid in (-1, 3, float('nan')):
            with self.assertRaises(ValueError):
                self.client.set_trim(invalid)

    def test_cross_axis_configuration_readback_and_reset_without_movement(self):
        self.client.connect()
        self.client.wait_until_connected(4)
        self.assertEqual(self.client.trim_matrix, ((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)))
        self.client.set_trim(((1.1,0.,.1),(.02,1.,0.),(-.04,0.,1.)))
        self.assertAlmostEqual(self.client.trim_matrix[0][0],1.1,places=6)
        self.assertAlmostEqual(self.client.trim_matrix[1][0],.02,places=6)
        self.assertAlmostEqual(self.client.trim_matrix[2][0],-.04,places=6)
        self.assertAlmostEqual(self.client.trim_matrix[0][2],.1,places=6)
        self.assertTrue(any(e['event']=='trim_matrix' for e in self.events()))
        self.client.set_trim()
        self.assertEqual(self.client.trim_matrix,((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)))
        self.client.stop()
        self.assertTrue(all(e['data']==[0,0,0] for e in self.events() if e['event']=='write'))
        for invalid in (float('nan'),float('inf'),2.1,-2.1):
            with self.assertRaises(ValueError):
                self.client.set_trim(((1.,0.,0.),(0.,1.,0.),(invalid,0.,1.)))

    def test_stored_trim_controller_and_live_status_are_readable(self):
        self.client.connect()
        self.client.wait_until_connected(4)
        self.assertEqual(self.client.read_trim(),
                         ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)))
        controller = self.client.read_controller()
        self.assertAlmostEqual(controller['kp_dist'], 4.)
        updated = self.client.set_controller({'horizon': .15})
        self.assertAlmostEqual(updated['horizon'], .15)
        self.assertTrue(any(e['event'] == 'controller_config' for e in self.events()))
        status = self.client.read_status()
        self.assertEqual(status.mode_name, 'idle')
        self.assertEqual(status.fault_name, 'none')
        self.assertTrue(status.wifi_connected)
        self.assertTrue(status.pose_fresh)
        self.assertEqual(status.packet_count, 7)
        self.assertAlmostEqual(status.angle, 90.)

    def test_controller_configuration_rejects_invalid_values_before_ble(self):
        self.client.connect()
        self.client.wait_until_connected(4)
        for update in ({'max_speed': 1.1}, {'velocity_alpha': 0},
                       {'unknown': 1}, {'horizon': float('nan')}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.client.set_controller(update)

    def test_rejected_configuration_writes_are_reported(self):
        with patch.dict(os.environ, {'BLE_TEST_REJECT_CONFIG': '1'}):
            self.client.connect()
        self.client.wait_until_connected(4)
        with self.assertRaisesRegex(RuntimeError, 'Firmware rejected trim'):
            self.client.set_trim(((.9, 0, 0), (0, 1, 0), (0, 0, 1)))
        with self.assertRaisesRegex(RuntimeError, 'Firmware rejected controller'):
            self.client.set_controller({'horizon': .15})

    def test_reconnect_telemetry_stops_before_command(self):
        self.client.connect()
        self.client.wait_until_connected(4)
        self.client.set_velocity({'x': .3})
        self.client.reconnect_telemetry()
        events = self.events()
        index = next(i for i, event in enumerate(events) if event['event'] == 'control')
        self.assertEqual(events[index]['data'], [2])
        self.assertEqual(events[index-1]['data'], [0, 0, 0])

    def test_restart_reconnects_without_resuming_motion(self):
        self.client.connect()
        self.client.wait_until_connected(4)
        self.client.set_velocity({'x': .3})
        self.client.restart_robot(timeout=4)
        self.assertTrue(self.client.is_connected)
        events = self.events()
        index = next(i for i, event in enumerate(events) if event['event'] == 'control')
        self.assertEqual(events[index]['data'], [3])
        self.assertEqual(events[index-1]['data'], [0, 0, 0])
        second_ready = [i for i, e in enumerate(events) if e['event'] == 'ready'][1]
        self.assertEqual(next(e['data'] for e in events[second_ready:] if e['event'] == 'write'), [0, 0, 0])

    def test_old_firmware_does_not_silently_accept_recovery(self):
        with patch.dict(os.environ, {'BLE_TEST_OLD_CONTROL': '1'}):
            self.client.connect()
        self.client.wait_until_connected(4)
        with self.assertRaisesRegex(RuntimeError, 'did not accept'):
            self.client.reconnect_telemetry()

    def test_old_firmware_can_drive_but_configuration_requires_matrix_support(self):
        with patch.dict(os.environ,{'BLE_TEST_LEGACY_TRIM':'1'}):
            self.client.connect()
        self.client.wait_until_connected(4)
        self.assertIsNone(self.client.trim_matrix)
        with self.assertRaisesRegex(RuntimeError,'does not support matrix trim'):
            self.client.set_trim()
        self.assertFalse(any(e['event']=='trim_matrix' for e in self.events()))

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

    def test_waypoint_uses_onboard_mode_and_lease_heartbeats(self):
        self.client.connect()
        self.client.wait_until_connected(4)
        self.assertEqual(self.client.set_waypoint(.2, .75, 90), (.2, .75, 90.))
        self.wait_for(lambda: any(len(e.get('data', ())) == 12 for e in self.events()))
        self.wait_for(lambda: len([e for e in self.events()
                                  if e.get('data') == [0]]) >= 2)
        self.client.stop()
        self.assertEqual([e for e in self.events() if e['event'] == 'write'][-1]['data'],
                         [0, 0, 0])

    def test_invalid_waypoint_never_reaches_worker(self):
        self.client.connect()
        self.client.wait_until_connected(4)
        for target in ((1, 0, 0), (0, 2, 0), (float('nan'), 0, 0)):
            with self.assertRaises(ValueError):
                self.client.set_waypoint(*target)
        self.assertFalse(any(len(e.get('data', ())) == 12 for e in self.events()))

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
from vsss._ble_client import MecanumBLEClient
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
    def gatts_set_buffer(self, handle, size): self.log.append(('buffer', handle, size))
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

    def test_long_trim_characteristic_reserves_the_full_buffer(self):
        self.server.control_callbacks={'matrix':lambda data:None}
        self.server.initial_values={'matrix':bytes(36)}
        self.server.start()
        self.assertIn(('buffer',1,36),self.radio.log)

    def test_empty_configuration_write_can_restore_canonical_value(self):
        self.server.control_callbacks['velocity']=lambda data:b'canonical'
        self.radio.gatts_write=unittest.mock.Mock()
        self.radio.payload=b''
        self.radio.handler(1,(7,0,b''))
        self.radio.handler(3,(7,1))
        self.radio.gatts_write.assert_called_once_with(1,b'canonical')

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
