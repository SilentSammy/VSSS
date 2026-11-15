import types
import unittest
from unittest.mock import Mock, patch

from mp.telemetry_recovery import TelemetryFreshness, TelemetryRecovery


class TelemetryFreshnessTests(unittest.TestCase):
    def setUp(self):
        self.clock = types.SimpleNamespace(ticks_diff=lambda a, b: a - b)
        self.time_patch = patch('mp.telemetry_recovery.time', self.clock)
        self.time_patch.start()
        self.addCleanup(self.time_patch.stop)

    def test_short_loss_pauses_then_recovers(self):
        guard = TelemetryFreshness(2000)
        self.assertEqual(guard.update(True, 100), ('fresh', 0))
        self.assertEqual(guard.update(False, 600), ('pause', 0))
        self.assertEqual(guard.update(False, 1500), ('wait', 900))
        self.assertEqual(guard.update(True, 1700), ('recover', 1100))
        self.assertEqual(guard.update(True, 1800), ('fresh', 0))

    def test_long_loss_becomes_terminal_and_reset_allows_retry(self):
        guard = TelemetryFreshness(2000)
        guard.update(False, 100)
        self.assertEqual(guard.update(False, 2100), ('fault', 2000))
        guard.reset()
        self.assertEqual(guard.update(False, 2200), ('pause', 0))


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
        self.wifi.config.assert_called_with(pm=0)
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
