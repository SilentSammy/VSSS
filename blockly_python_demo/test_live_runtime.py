"""Exercise the live supervisor/worker protocol with fake hardware, never BLE."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from threading import Event, Thread
import time
import unittest
from unittest.mock import patch

from live_runtime import LiveRunner
from server import validate


@dataclass
class Pose:
    x: float = .1
    y: float = .2
    angle: float = 90.


@dataclass
class Status:
    mode: int = 0
    flags: int = 3
    fault: int = 0
    reserved: int = 0
    x: float = .1
    y: float = .2
    angle: float = 90.
    packet_count: int = 10


class FakeCar:
    def __init__(self, number):
        self.number = number
        self.pose = Pose()
        self.is_connected = True
        self.connection_generation = 1
        self.events = [('connect', number)]
        self.moving = Event()
        self.gps = self

    @property
    def position(self):
        return None if self.pose is None else (self.pose.x, self.pose.y)

    def set_velocity(self, x=0., y=0., w=0.):
        if not all(-1 <= v <= 1 for v in (x, y, w)):
            raise ValueError('velocity must be within [-1, 1]')
        self.events.append(('velocity', x, y, w))
        self.moving.set()

    def go_to(self, x, y, angle):
        self.events.append(('waypoint', x, y, angle))
        self.moving.set()
        return x, y, angle

    def read_status(self):
        self.events.append(('status',))
        return Status()

    def stop(self):
        self.events.append(('stop',))

    def close(self):
        self.events.append(('close',))
        self.is_connected = False


PREFIX = 'from car import Car\nfrom time import sleep\nwith Car(1) as car:\n'


class LiveTests(unittest.TestCase):
    def setUp(self):
        self.cars = []

        def factory(number):
            car = FakeCar(number)
            self.cars.append(car)
            return car

        self.factory = factory
        self.runner = LiveRunner(factory, lease_seconds=2, max_seconds=5)

    def tearDown(self):
        self.runner.shutdown()

    def start(self, body):
        code = PREFIX + body
        validate(code)
        return self.runner.start(code)['run_id']

    def finish(self, run_id, renew=True, timeout=4):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self.runner.status(run_id if renew else None)
            if result['state'] not in {'running', 'stopping'}:
                return result
            time.sleep(.02)
        self.fail('Run did not finish: ' + str(self.runner.status()))

    def wait_car(self):
        deadline = time.monotonic() + 2
        while not self.cars:
            if time.monotonic() > deadline:
                self.fail('No connection attempt')
            time.sleep(.01)
        return self.cars[0]

    def assert_closed(self):
        events = self.cars[0].events
        self.assertIn(('stop',), events)
        self.assertIn(('close',), events)
        last_motion = max((i for i, e in enumerate(events) if e[0] in {'velocity', 'waypoint'}), default=-1)
        self.assertGreater(events.index(('stop',)), last_motion)

    def test_live_commands_output_and_real_duration(self):
        began = time.monotonic()
        run = self.start('    car.wait_for_position(timeout=1)\n'
                         '    print(car.pose)\n'
                         '    print(car.position)\n'
                         '    car.set_velocity(x=.2, y=0, w=0)\n'
                         '    sleep(.15)\n'
                         '    car.go_to(0, .5, 90)\n'
                         '    print(car.read_status())\n')
        result = self.finish(run)
        self.assertEqual(result['state'], 'finished', result)
        self.assertIn('Pose(x=0.1, y=0.2', result['output'])
        self.assertIn('(0.1, 0.2)', result['output'])
        self.assertGreaterEqual(time.monotonic() - began, .15)
        self.assertIn(('velocity', .2, 0, 0), self.cars[0].events)
        self.assertIn(('waypoint', 0, .5, 90), self.cars[0].events)
        self.assert_closed()

    def test_stop_interrupts_long_sleep_and_no_later_commands(self):
        run = self.start('    car.set_velocity(.2, 0, 0)\n    sleep(60)\n    car.go_to(0, .5, 90)\n')
        self.assertTrue(self.wait_car().moving.wait(2))
        self.runner.stop(run)
        self.assertEqual(self.finish(run)['state'], 'stopped')
        self.assertFalse(any(e[0] == 'waypoint' for e in self.cars[0].events))
        self.assert_closed()

    def test_stop_interrupts_cpu_loop(self):
        run = self.start('    car.set_velocity(.2, 0, 0)\n    while True:\n        pass\n')
        self.assertTrue(self.wait_car().moving.wait(2))
        self.runner.stop(run)
        self.assertEqual(self.finish(run)['state'], 'stopped')
        self.assert_closed()

    def test_error_stops_and_disconnects(self):
        run = self.start('    car.set_velocity(.2, 0, 0)\n    print(1 / 0)\n')
        result = self.finish(run)
        self.assertEqual(result['state'], 'failed')
        self.assertIn('ZeroDivisionError', result['error'])
        self.assert_closed()

    def test_browser_lease_and_run_time_limit(self):
        for lease, limit, expected in ((.3, 5, 'Editor disconnected'), (2, .4, 'Run time limit')):
            with self.subTest(expected=expected):
                self.cars.clear()
                self.runner = LiveRunner(self.factory, lease_seconds=lease, max_seconds=limit)
                run = self.start('    car.set_velocity(.2, 0, 0)\n    sleep(60)\n')
                result = self.finish(run, renew=False)
                self.assertEqual(result['state'], 'stopped', result)
                self.assertIn(expected, result['reason'])
                self.assert_closed()

    def test_missing_pose_blocks_motion_and_stale_pose_stops_motion(self):
        def unavailable(number):
            car = self.factory(number)
            car.pose = None
            return car
        self.runner = LiveRunner(unavailable)
        run = self.start('    car.set_velocity(.2, 0, 0)\n')
        self.assertEqual(self.finish(run)['state'], 'failed')
        self.assertFalse(self.cars[0].moving.is_set())
        self.cars.clear()
        self.runner = LiveRunner(self.factory)
        run = self.start('    car.set_velocity(.2, 0, 0)\n    sleep(60)\n')
        car = self.wait_car()
        self.assertTrue(car.moving.wait(2))
        car.pose = None
        result = self.finish(run)
        self.assertIn('stale', result['reason'])
        self.assert_closed()

    def test_lost_connection_and_reconnection_end_run(self):
        for changed in ('is_connected', 'connection_generation'):
            with self.subTest(changed=changed):
                self.cars.clear()
                run = self.start('    car.set_velocity(.2, 0, 0)\n    sleep(60)\n')
                car = self.wait_car()
                self.assertTrue(car.moving.wait(2))
                setattr(car, changed, False if changed == 'is_connected' else 2)
                self.assertEqual(self.finish(run)['state'], 'stopped')
                self.assert_closed()

    def test_busy_and_stale_stop_do_not_replace_active_run(self):
        run = self.start('    sleep(60)\n')
        with self.assertRaises(ValueError):
            self.runner.start('print(1)')
        self.runner.stop('old-run-id')
        self.assertEqual(self.runner.status(run)['state'], 'running')
        self.runner.stop(run)
        self.assertEqual(self.finish(run)['state'], 'stopped')

    def test_stop_during_connection_never_allows_motion(self):
        connecting, release = Event(), Event()

        def slow_factory(number):
            connecting.set()
            release.wait(2)
            return self.factory(number)

        self.runner = LiveRunner(slow_factory)
        run = self.start('    car.set_velocity(.2, 0, 0)\n')
        self.assertTrue(connecting.wait(2))
        self.runner.stop(run)
        release.set()
        self.assertEqual(self.finish(run)['state'], 'stopped')
        self.assertFalse(self.cars[0].moving.is_set())
        self.assert_closed()

    def test_camera_wait_is_cancellable(self):
        def unavailable(number):
            car = self.factory(number)
            car.pose = None
            return car
        self.runner = LiveRunner(unavailable)
        run = self.start('    car.wait_for_position(timeout=60)\n')
        self.wait_car()
        self.runner.stop(run)
        self.assertEqual(self.finish(run)['state'], 'stopped')
        self.assert_closed()

    def test_output_limit_and_stop_ack_failure_are_visible(self):
        run = self.start('    for i in range(20000):\n        print("hello")\n')
        self.assertEqual(self.finish(run)['state'], 'failed')
        self.assert_closed()
        self.cars.clear()

        def failing_stop(number):
            car = self.factory(number)
            def stop():
                raise TimeoutError('fake acknowledgement timeout')
            car.stop = stop
            return car

        self.runner = LiveRunner(failing_stop)
        run = self.start('    print(car.pose)\n')
        result = self.finish(run)
        self.assertEqual(result['state'], 'failed')
        self.assertIn('Stop was not acknowledged', result['error'])
        self.assertIn(('close',), self.cars[0].events)

    def test_validation_rejects_other_imports_and_unguarded_constructor(self):
        for code in ('import os', 'from car import GPS', 'Car(1)', 'car.restart()',
                     'print(car._client)', 'car.pose = 1'):
            with self.subTest(code=code), self.assertRaises(ValueError):
                validate(code)


class RealAPIPipelineTests(unittest.TestCase):
    def test_live_supervisor_through_real_car_and_ble_worker_with_fake_radio(self):
        root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(root))
        from gps import GPS
        # The actual UDP receiver is restricted to localhost test packets and
        # passive mode; the radio substitute is verified before constructing Car.
        gps = GPS(port=0, subscribe=False)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        stop_sending = Event()
        port = gps.port
        packet = json.dumps(dict(board_detected=True, markers=[dict(id=1, x=.1, y=.2, angle=90)])).encode()

        def publish():
            while not stop_sending.is_set():
                sender.sendto(packet, ('127.0.0.1', port))
                stop_sending.wait(.03)

        publisher = Thread(target=publish, daemon=True)
        runner = LiveRunner(lease_seconds=2, max_seconds=5)
        try:
            with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
                    'PYTHONPATH': str(root / 'tests/fake_ble') + os.pathsep + str(root),
                    'BLE_TEST_LOG': str(Path(folder) / 'ble.jsonl')}):
                radio = subprocess.check_output([sys.executable, '-c', 'import bleak; print(bleak.__file__)'],
                                                cwd=root, text=True).strip()
                self.assertEqual(Path(radio).resolve(), (root / 'tests/fake_ble/bleak.py').resolve())
                publisher.start()
                run_id = runner.start(PREFIX + '    car.wait_for_position(timeout=1)\n'
                                      '    car.set_velocity(.2, 0, 0)\n'
                                      '    sleep(.2)\n'
                                      '    print(car.read_status())\n')['run_id']
                deadline = time.monotonic() + 6
                while time.monotonic() < deadline:
                    result = runner.status(run_id)
                    if result['state'] not in {'running', 'stopping'}:
                        break
                    time.sleep(.02)
                self.assertEqual(result['state'], 'finished', result)
                self.assertIn('RobotStatus(', result['output'])
                events = [json.loads(line) for line in (Path(folder) / 'ble.jsonl').read_text().splitlines()]
                self.assertTrue(any(e['event'] == 'scan' and e['name'] == 'car1' for e in events))
                moving = [i for i, e in enumerate(events) if e['event'] == 'write' and e['data'] == [25, 0, 0]]
                self.assertTrue(moving, events)
                self.assertTrue(any(e['event'] == 'write' and e['data'] == [0, 0, 0]
                                    for e in events[max(moving) + 1:]))
                self.assertTrue(any(e['event'] == 'disconnect' for e in events))
        finally:
            runner.shutdown()
            stop_sending.set()
            if publisher.ident:
                publisher.join(timeout=1)
            sender.close()
            gps.close()


if __name__ == '__main__':
    unittest.main()
