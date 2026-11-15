import json
from pathlib import Path
import tempfile
import time
import unittest

from vsss._bag import BagReader


class ProviderSourceTests(unittest.TestCase):
    def make_bag(self, directory):
        path = Path(directory) / 'session.jsonl'
        rows = [
            {'type': 'session_start', 'monotonic_ns': 900},
            {'type': 'provider_snapshot', 'monotonic_ns': 950,
             'controlled_cars': ['car1'], 'vision': {'local_port': 5001}},
            {'type': 'udp', 'monotonic_ns': 1_000_000_000, 'sequence': 1, 'valid': True},
            {'type': 'ble', 'monotonic_ns': 1_050_000_000, 'request': 2},
            {'type': 'udp', 'monotonic_ns': 1_100_000_000, 'sequence': 3, 'valid': True},
        ]
        path.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
        return path

    def test_bag_provider_filters_topics_and_replays_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = BagReader(self.make_bag(directory), ('udp',), speed=1)
            self.assertEqual(provider.snapshot['controlled_cars'], ['car1'])
            self.assertEqual(provider.read(0)['sequence'], 1)
            self.assertIsNone(provider.read(.01))
            started = time.monotonic()
            self.assertEqual(provider.read(.2)['sequence'], 3)
            self.assertGreaterEqual(time.monotonic() - started, .07)
            self.assertTrue(provider.finished)
            provider.close()

    def test_empty_bag_and_close_finish_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'empty.jsonl'
            path.write_text('{"type":"session_start"}\n', encoding='utf-8')
            provider = BagReader(path, ('udp',), speed=10)
            self.assertTrue(provider.finished)
            self.assertIsNone(provider.read(0))
            provider.close()

    def test_reader_rejects_invalid_playback_speed(self):
        with tempfile.TemporaryDirectory() as directory:
            for speed in (0, -1, float('nan')):
                with self.subTest(speed=speed), self.assertRaises(ValueError):
                    BagReader(self.make_bag(directory), speed=speed)


if __name__ == '__main__':
    unittest.main()
