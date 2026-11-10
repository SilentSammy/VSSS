"""Client UDP protocol checks, independent of the external vision service."""
import json
import socket
import time
import unittest

from gps import GPS


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        # Reserve the adjacent protocol port before starting the receiver.
        for _ in range(100):
            self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.server.bind(('127.0.0.1', 0))
            port = self.server.getsockname()[1] - 1
            try:
                self.gps = GPS(port=port)
                break
            except OSError:
                self.server.close()
        else:
            self.fail('Could not reserve adjacent UDP ports')
        self.server.settimeout(3)
        self.target = ('127.0.0.1', self.gps.port)
        self.packet = dict(board_detected=True,
                           markers=[dict(id=1, x=.1, y=.2, angle=90)], balls=[])

    def tearDown(self):
        self.gps.close()
        self.server.close()

    def send_pose(self):
        self.server.sendto(json.dumps(self.packet).encode(), self.target)
        deadline = time.monotonic() + 1
        while self.gps.get_car(1) is None:
            if time.monotonic() > deadline:
                self.fail('No unicast pose')
            time.sleep(.005)

    def test_subscription_renews_and_old_broadcast_cannot_replace_unicast(self):
        self.send_pose()
        self.assertEqual(self.gps._server, self.server.getsockname())
        data, address = self.server.recvfrom(65535)
        self.assertEqual(data, b'VSSS_SUBSCRIBE')
        self.assertEqual(address[1], self.gps.port)
        self.send_pose()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as legacy:
            legacy.sendto(json.dumps(dict(self.packet, markers=[])).encode(), self.target)
            time.sleep(.03)
        self.assertIsNotNone(self.gps.get_car(1))
        data, _ = self.server.recvfrom(65535)
        self.assertEqual(data, b'VSSS_SUBSCRIBE')

    def test_close_unsubscribes(self):
        self.send_pose()
        self.gps.close()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            data, address = self.server.recvfrom(65535)
            if data == b'VSSS_UNSUBSCRIBE':
                self.assertEqual(address[1], self.gps.port)
                return
        self.fail('No unsubscribe datagram')

    def test_receiver_recovers_after_server_restart(self):
        self.send_pose()
        address = self.server.getsockname()
        self.server.close()
        time.sleep(1.2)
        self.assertTrue(self.gps._thread.is_alive())
        self.assertIsNone(self.gps.get_car(1))
        self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server.bind(address)
        self.server.settimeout(3)
        data, _ = self.server.recvfrom(65535)
        self.assertEqual(data, b'VSSS_SUBSCRIBE')
        self.send_pose()
        self.assertIsNotNone(self.gps.get_car(1))
