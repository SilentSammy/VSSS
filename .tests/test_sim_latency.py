"""Transmission delays preserve progressing physics and expose delayed onboard poses."""
import unittest
import time

from vsss._sim import SimWorld, load_config
from vsss._sim_provider import SimProvider


class SimulationLatencyTests(unittest.TestCase):
    def test_onboard_feedback_waits_for_udp_delivery(self):
        config = load_config()
        config['latency'] = dict(udp_ms=200, ble_ms=0)
        world = SimWorld(config)
        car = world.cars[1]
        car.command = (.5, 0, 0)
        for _ in range(30):
            world.step(.005)
        self.assertFalse(world.status(car)[1] & 2)
        for _ in range(30):
            world.step(.005)
        status = world.status(car)
        self.assertTrue(status[1] & 2)
        self.assertLess(status[5], car.y - .02)

    def test_zero_latency_exposes_current_pose(self):
        config = load_config()
        config['latency'] = dict(udp_ms=0, ble_ms=0)
        world = SimWorld(config)
        car = world.cars[1]
        car.command = (.5, 0, 0)
        for _ in range(60):
            world.step(.005)
        self.assertAlmostEqual(world.status(car)[5], car.y, delta=.012)

    def test_ble_request_delay_does_not_pause_physics(self):
        config = load_config()
        config['latency'] = dict(udp_ms=100, ble_ms=250)
        provider = SimProvider(config)
        provider.on_listening()
        session = provider.create_car_session('car1', lambda *args, **kwargs: None)
        client, queue = session.attach()
        queue.get(timeout=1)  # Initial ready/cache message.
        try:
            start = provider.world.time
            session.command(client, dict(velocity=[64, 0, 0], sequence=1))
            until = time.monotonic() + 1
            while time.monotonic() < until:
                if queue.get(timeout=1).get('written') == 1:
                    break
            self.assertGreater(provider.world.time - start, .12)
            self.assertEqual(provider.world.cars[1].mode, 1)
        finally:
            provider.closed.set()
            provider.thread.join(timeout=2)
            session.close()

    def test_disconnect_and_edit_cancel_commands_still_in_transit(self):
        config = load_config()
        config['latency']['ble_ms'] = 150
        provider = SimProvider(config)
        session = provider.create_car_session('car1', lambda *args, **kwargs: None)
        provider.controllers['car1'] = session
        first, _ = session.attach()
        second, _ = session.attach()
        try:
            session.command(first, dict(velocity=[64, 0, 0], sequence=1))
            session.detach(first)
            time.sleep(.2)
            self.assertEqual(provider.world.cars[1].mode, 0)
            session.command(second, dict(velocity=[64, 0, 0], sequence=1))
            provider.edit('editor', dict(operation='move', kind='car', id=1, x=.3, y=.3))
            time.sleep(.2)
            self.assertEqual(provider.world.cars[1].mode, 0)
            self.assertEqual(provider.world.cars[1].command, (0, 0, 0))
        finally:
            session.close()

    def test_invalid_latency_settings_are_rejected(self):
        for latency in (dict(udp_ms=-1, ble_ms=0), dict(udp_ms=100, ble_ms=float('nan')),
                        dict(udp_ms=0), dict(udp_ms=500, ble_ms=0)):
            with self.subTest(latency=latency), self.assertRaises(ValueError):
                SimWorld(dict(load_config(), latency=latency))


if __name__ == '__main__':
    unittest.main()
