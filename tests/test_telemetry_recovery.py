import types
import unittest
from unittest.mock import Mock, patch

from mp.telemetry_recovery import TelemetryRecovery


class TelemetryRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.now = 0
        self.clock = types.SimpleNamespace(ticks_ms=lambda: self.now,
                                           ticks_add=lambda a, b: a + b,
                                           ticks_diff=lambda a, b: a - b)
        self.time_patch = patch('mp.telemetry_recovery.time', self.clock)
        self.time_patch.start()
        self.addCleanup(self.time_patch.stop)
        self.wifi = Mock()
        self.wifi.isconnected.return_value = False
        self.factory = Mock()
        self.settings = dict(wifi_ssid='lab', wifi_password='secret', car_id=1,
                             telemetry_host='192.0.2.1', telemetry_port=5000)
        self.recovery = TelemetryRecovery(self.wifi, self.settings, self.factory)

    def test_retries_without_blocking_and_creates_receiver_after_association(self):
        self.recovery.poll()
        self.wifi.connect.assert_called_once_with('lab', 'secret')
        self.recovery.poll()
        self.assertEqual(self.wifi.connect.call_count, 1)
        self.wifi.isconnected.return_value = True
        self.recovery.poll()
        self.factory.assert_called_once_with(1, server='192.0.2.1', port=5000, subscribe=True)
        self.factory.return_value.poll.assert_called_once()

    def test_explicit_reconnect_closes_socket_and_restarts_wifi(self):
        self.wifi.isconnected.return_value = True
        self.recovery.poll()
        receiver = self.factory.return_value
        self.recovery.reconnect()
        receiver.close.assert_called_once()
        self.wifi.disconnect.assert_called_once()
        self.wifi.isconnected.return_value = False
        self.recovery.poll()
        self.wifi.connect.assert_called_once()

    def test_socket_failure_retries_after_interval(self):
        self.wifi.isconnected.return_value = True
        self.factory.side_effect = [OSError('busy'), Mock()]
        self.recovery.poll()
        self.recovery.poll()
        self.assertEqual(self.factory.call_count, 1)
        self.now = 10000
        self.recovery.poll()
        self.assertEqual(self.factory.call_count, 2)
