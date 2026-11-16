import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools.vsss_plot import plotter_from_snapshot, main


class PlotToolTests(unittest.TestCase):
    def test_persist_reconnects_after_end_and_disconnect_with_fresh_metadata(self):
        clients = [Mock(snapshot={'feed': index}, finished=False) for index in range(3)]
        def end_stream(timeout):
            clients[0].finished = True
            return {'type': 'stream_end', 'complete': True}
        clients[0].read.side_effect = end_stream
        clients[1].read.side_effect = ConnectionError('provider stopped')
        packet = {'board_detected': True, 'markers': [], 'balls': []}
        clients[2].read.return_value = {'type': 'udp', 'valid': True, 'packet': packet}
        with tempfile.TemporaryDirectory() as directory, \
                patch('tools.vsss_plot.provider_from_args', side_effect=clients) as connect, \
                patch('tools.vsss_plot.plotter_from_snapshot',
                      side_effect=lambda snapshot, width, fallback:
                          plotter_from_snapshot({}, width, fallback)) as make_plotter:
            output = Path(directory) / 'field.png'
            main(['--persist', '--duration', '1.4', '--width', '320',
                  '--field-width', '1.6', '--field-height', '3', '--output', str(output)])
            self.assertTrue(output.is_file())
            self.assertEqual(connect.call_count, 3)
            self.assertEqual([call.args[0] for call in make_plotter.call_args_list],
                             [client.snapshot for client in clients])
        for client in clients:
            client.close.assert_called_once()

    def test_default_disconnect_exits_without_reconnecting(self):
        client = Mock(snapshot={}, finished=False)
        client.read.side_effect = ConnectionError('provider stopped')
        with tempfile.TemporaryDirectory() as directory, \
                patch('tools.vsss_plot.provider_from_args', return_value=client) as connect:
            with self.assertRaisesRegex(ConnectionError, 'provider stopped'):
                main(['--field-width', '1.6', '--field-height', '3',
                      '--output', str(Path(directory) / 'field.png')])
            connect.assert_called_once()
        client.close.assert_called_once()

    def test_renderer_places_marker_and_ball_on_field(self):
        plotter = plotter_from_snapshot({}, width=320, fallback=(1.6, 3.0))
        frame = plotter.render_packet({'board_detected': True,
                                'markers': [{'id': 1, 'x': 0, 'y': .5, 'angle': 90}],
                                'balls': [{'x': .2, 'y': -.3}]})
        self.assertEqual(frame.shape[1], 320)
        self.assertGreater(frame.shape[0], frame.shape[1])
        self.assertFalse((frame == plotter.background).all())

    def test_headless_bag_replay_saves_last_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            bag = Path(directory) / 'bag.jsonl'
            output = Path(directory) / 'field.png'
            rows = [
                {'type': 'provider_snapshot', 'monotonic_ns': 1,
                 'controlled_cars': [], 'vision': {}},
                {'type': 'udp', 'monotonic_ns': 10, 'valid': True,
                 'packet': {'board_detected': True,
                            'markers': [{'id': 1, 'x': .1, 'y': .2, 'angle': 45}],
                            'balls': [{'x': -.2, 'y': .3}]}}
            ]
            bag.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
            from vsss._bag import BagReader
            with contextlib.redirect_stdout(io.StringIO()) as stdout, \
                    patch('tools.vsss_plot.provider_from_args', return_value=BagReader(bag, speed=1000)):
                main(['--width', '320',
                      '--field-width', '1.6', '--field-height', '3.0',
                      '--output', str(output)])
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1000)
            self.assertIn('Saved', stdout.getvalue())


if __name__ == '__main__':
    unittest.main()
