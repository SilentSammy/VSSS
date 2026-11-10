"""Real sockets: automatic renewal, multiple subscribers, cleanup and stale broadcasts."""
import json
import socket
import time
import unittest

from detection.telemetry import TelemetryPublisher, SUBSCRIBE, UNSUBSCRIBE
from gps import GPS


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.gps = GPS(port=0)
        self.publisher = TelemetryPublisher(None, self.gps.port)
        self.target = ('127.0.0.1', self.gps.port + 1)
        self.packet = dict(board_detected=True,
                           markers=[dict(id=1, x=.1, y=.2, angle=90)], balls=[])

    def tearDown(self):
        self.gps.close()
        self.publisher.close()

    def receive_pose(self):
        deadline = time.monotonic() + 1
        while self.gps.get_car(1) is None:
            if time.monotonic() > deadline:
                self.fail('No unicast pose')
            time.sleep(.005)

    def test_subscription_renews_and_old_broadcast_cannot_replace_unicast(self):
        # Seed discovery over loopback; subsequent renewals use the learned server.
        self.gps._socket.sendto(SUBSCRIBE, self.target)
        self.publisher.send(self.packet)
        self.receive_pose()
        self.assertEqual(self.gps._server, self.target)
        address = next(iter(self.publisher._subscribers))
        first_expiry = self.publisher._subscribers[address]
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as legacy:
            old = dict(self.packet, markers=[])
            legacy.sendto(json.dumps(old).encode(), ('127.0.0.1', self.gps.port))
            time.sleep(.03)
        self.assertIsNotNone(self.gps.get_car(1))
        time.sleep(1.1)
        self.publisher.send(self.packet)
        self.assertGreater(self.publisher._subscribers[address], first_expiry)
        self.gps.close()
        self.publisher.send(self.packet)
        self.assertFalse(self.publisher._subscribers)

    def test_multiple_students_and_expiry(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as other:
            other.bind(('127.0.0.1', 0))
            other.settimeout(1)
            other.sendto(SUBSCRIBE, self.target)
            self.gps._socket.sendto(SUBSCRIBE, self.target)
            self.publisher.send(self.packet)
            self.receive_pose()
            self.assertEqual(json.loads(other.recvfrom(65535)[0]), self.packet)
            self.assertEqual(len(self.publisher._subscribers), 2)
            address = other.getsockname()
            self.publisher._subscribers[address] = time.monotonic() - 1
            self.publisher.send(self.packet)
            self.assertNotIn(address, self.publisher._subscribers)
            other.sendto(UNSUBSCRIBE, self.target)
            self.publisher.send(self.packet)

    def test_receiver_recovers_after_server_restart(self):
        self.gps._socket.sendto(SUBSCRIBE, self.target)
        self.publisher.send(self.packet)
        self.receive_pose()
        self.publisher.close()
        time.sleep(1.2)
        self.assertTrue(self.gps._thread.is_alive())
        self.assertIsNone(self.gps.get_car(1))
        self.publisher = TelemetryPublisher(None, self.gps.port)
        deadline = time.monotonic() + 3
        while self.gps.get_car(1) is None and time.monotonic() < deadline:
            self.publisher.send(self.packet)
            time.sleep(.02)
        self.assertIsNotNone(self.gps.get_car(1))
