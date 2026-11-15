"""Unique-frame freshness and invalid-packet isolation on the robot."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from mp.pose_receiver import PoseReceiver


class PoseReceiverTests(unittest.TestCase):
    def setUp(self):
        self.now = 1000
        self.clock = SimpleNamespace(ticks_ms=lambda: self.now, ticks_add=lambda a,b:a+b,
                                     ticks_diff=lambda a,b:a-b)
        self.socket = Mock()
        self.pending = []
        def receive(*args):
            if self.pending:
                return self.pending.pop(0), ('192.168.12.1', 5001)
            raise BlockingIOError()
        self.socket.recvfrom.side_effect = receive
        self.time_patch = patch('mp.pose_receiver.time', self.clock)
        self.socket_patch = patch('mp.pose_receiver.socket.socket', return_value=self.socket)
        self.time_patch.start()
        self.socket_patch.start()
        self.addCleanup(self.time_patch.stop)
        self.addCleanup(self.socket_patch.stop)
        self.receiver = PoseReceiver(1)
        self.addCleanup(self.receiver.close)

    def send(self, sequence=1, stream='first', age=100, x=.1):
        self.pending.append(json.dumps(dict(board_detected=True, stream_id=stream,
            frame_sequence=sequence, frame_age_ms=age,
            markers=[dict(id=1, x=x, y=.2, angle=90)])).encode())
        return self.receiver.poll()

    def test_frame_age_counts_toward_freshness(self):
        self.send(age=140)
        self.assertTrue(self.receiver.fresh)
        self.now += 400
        self.assertEqual(self.receiver.age_ms, 540)
        self.assertFalse(self.receiver.fresh)

    def test_repeated_and_out_of_order_frames_do_not_refresh_pose(self):
        self.send(sequence=10)
        self.now += 300
        self.assertFalse(self.send(sequence=10, x=9))
        self.assertFalse(self.send(sequence=9, x=9))
        self.assertEqual(self.receiver.received_ms, 1000)
        self.assertEqual(self.receiver.pose[0], .1)
        self.now += 101
        self.assertFalse(self.receiver.fresh)

    def test_new_stream_can_restart_sequence(self):
        self.send(sequence=100)
        self.now += 100
        self.assertTrue(self.send(sequence=1, stream='restarted', x=.3))
        self.assertEqual(self.receiver.pose[0], .3)
        self.assertEqual(self.receiver.stream_id, 'restarted')
        self.assertTrue(self.receiver.fresh)

    def test_receive_and_sequence_gaps_are_recorded(self):
        self.send(sequence=10)
        self.now += 180
        self.send(sequence=13)
        self.assertEqual(self.receiver.receive_gap_ms, 180)
        self.assertEqual(self.receiver.sequence_gap, 3)
        self.now += 20
        self.send(sequence=2, stream='new')
        self.assertIsNone(self.receiver.receive_gap_ms)
        self.assertIsNone(self.receiver.sequence_gap)

    def test_invalid_pose_and_age_do_not_replace_valid_state(self):
        self.send()
        for age, x in ((-1, .2), (float('nan'), .2), (100, float('nan'))):
            self.now += 100
            self.assertFalse(self.send(sequence=2, age=age, x=x))
        self.assertEqual(self.receiver.received_ms, 1000)
        self.assertEqual(self.receiver.frame_sequence, 1)
        self.assertEqual(self.receiver.parse_errors, 3)

    def test_poll_drain_is_bounded(self):
        for sequence in range(20):
            self.pending.append(json.dumps(dict(board_detected=False, markers=[],
                frame_sequence=sequence, frame_age_ms=10, stream_id='first')).encode())
        self.receiver.poll()
        self.assertEqual(self.receiver.packets, 8)
        self.assertEqual(len(self.pending), 12)

    def test_receive_memory_pressure_invalidates_pose_and_can_recover(self):
        self.send(sequence=1)
        receive = self.socket.recvfrom.side_effect
        self.socket.recvfrom.side_effect = MemoryError('RX buffer')
        self.receiver.poll()
        self.assertFalse(self.receiver.fresh)
        self.assertEqual(self.receiver.memory_errors, 1)
        self.socket.recvfrom.side_effect = receive
        self.now += 100
        self.send(sequence=2)
        self.assertTrue(self.receiver.fresh)


if __name__ == '__main__':
    unittest.main()
