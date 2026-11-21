import contextlib
import io
from pathlib import Path
from queue import Queue
import subprocess
import sys
import tempfile
from threading import Thread
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from vsss import Velocity
from vsss.playback import VSSSPlayback
from vsss.plot import ProviderFeed, make_plotter, render_state
from vsss.simulation import VSSSSim
from vsss.teleop import assign_inputs, explicit_assignments, velocity, main as teleop


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate(): return
        time.sleep(.01)
    raise AssertionError('timed out')


class ToolsTest(unittest.TestCase):
    def test_teleop_assignments_and_keyboard_gamepad_mix(self):
        self.assertEqual(explicit_assignments(['3=wasd+gamepad:0', '8'], ['wasd', 'ijkl', 'gamepad:0']),
                         [(3, ('wasd', 'gamepad:0')), (8, ('ijkl',))])
        self.assertEqual(assign_inputs([1], ['wasd', 'ijkl', 'gamepad:0']), [(1, ('wasd', 'gamepad:0'))])
        for bad in (['1', '1'], ['1=wasd', '2=wasd'], ['1=gamepad:9'], ['1='], ['1=wasd+wasd']):
            with self.assertRaises(ValueError): explicit_assignments(bad, ['wasd', 'ijkl'])
        inp = Mock()
        inp.get_bipolar_ctrl.side_effect = [0, 1, 0, -1]
        self.assertEqual(velocity(('wasd',), inp), dict(x=.5, y=0, w=.5))
        inp.get_axis.assert_not_called()

    def test_teleop_50hz_changed_inputs_only_and_final_stop(self):
        provider = Mock()
        provider.describe.return_value = {'capabilities': {'commands': True}}
        inp = SimpleNamespace(KEYBOARD_AVAILABLE=True, pressed_keys=set())
        commands = [dict(x=x, y=0, w=0) for x in (0, .001, .1, .101)]
        waits = []
        def sleep(delay):
            waits.append(delay)
            if len(waits) == 4: raise KeyboardInterrupt
        with patch('vsss.teleop.VSSSProvider.connect', return_value=provider), \
             patch.dict(sys.modules, {'vsss.input': inp, 'inputs': SimpleNamespace(devices=SimpleNamespace(gamepads=[]))}), \
             patch('vsss.teleop.velocity', side_effect=commands), patch('vsss.teleop.time.sleep', side_effect=sleep), \
             contextlib.redirect_stdout(io.StringIO()):
            teleop(['--car', '1=wasd'])
        self.assertEqual(provider.cars.set_velocity.call_count, 2)
        self.assertEqual(provider.cars.set_velocity.call_args_list[0].args, (1, Velocity()))
        self.assertEqual(provider.cars.stop.call_count, 2)
        self.assertTrue(all(0 < delay <= .02 for delay in waits))
        provider.close.assert_called_once()

    def test_list_does_not_start_input(self):
        provider = Mock()
        provider.describe.return_value = {'capabilities': {'commands': True}}
        provider.cars.available.return_value = [1, 2]
        with patch('vsss.teleop.VSSSProvider.connect', return_value=provider), contextlib.redirect_stdout(io.StringIO()):
            teleop(['--list', '--refresh'])
        provider.cars.available.assert_called_once_with(refresh=True)
        provider.cars.stop.assert_not_called()

    def test_plot_reconnects_across_sim_and_playback(self):
        with tempfile.TemporaryDirectory() as directory:
            endpoint = Path(directory) / 'rpc.json'
            feed = ProviderFeed(endpoint)
            try:
                with VSSSSim().start() as sim, sim.serve(endpoint=endpoint, port=0):
                    wait_for(lambda: feed.snapshot()[2] == sim.source_id)
                    self.assertEqual(feed.snapshot()[1], 'simulation')
                with VSSSPlayback('examples/recording.jsonl', speed=10).start() as player, player.serve(endpoint=endpoint, port=0):
                    wait_for(lambda: feed.snapshot()[2] == player.source_id)
                    self.assertTrue(player.finished.wait(3))
                    wait_for(lambda: feed.snapshot()[1] == 'Playback finished')
                    self.assertEqual(feed.snapshot()[0].cars[0].pose.x, .1)
            finally:
                feed.close()
            self.assertFalse(feed.thread.is_alive())

    def test_original_renderer_and_headless_cli(self):
        from aruco_gps.plotter2d import BoardPlotter2D
        with tempfile.TemporaryDirectory() as directory, VSSSSim().start() as sim:
            plotter = make_plotter(sim.gps.state().field, 320)
            self.assertIsInstance(plotter, BoardPlotter2D)
            frame = render_state(plotter, sim.gps.state())
            self.assertEqual(frame.shape[1], 320)
            self.assertGreater(frame.std(), 1)
            endpoint = Path(directory) / 'rpc.json'
            output = Path(directory) / 'plot.png'
            with sim.serve(endpoint=endpoint, port=0):
                result = subprocess.run([sys.executable, '-m', 'vsss.plot', '--endpoint', str(endpoint),
                                         '--duration', '.8', '--width', '320', '--output', str(output)],
                                        capture_output=True, text=True, timeout=8)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertGreater(output.stat().st_size, 1000)

    def test_all_provider_mains_handle_pending_interrupt_and_cleanup(self):
        # interrupt_main queues the same Python SIGINT as Ctrl+C. On Windows an
        # indefinite Event.wait would never deliver it; the process would hang.
        code = '''
import _thread, importlib, sys
from threading import Timer
module = importlib.import_module(sys.argv.pop(1))
timer = Timer(.8, _thread.interrupt_main)
timer.daemon = True
timer.start()
module.main()
'''
        for module, extra in [('vsss.simulation', []), ('vsss.live', ['--vision-server', '']),
                              ('vsss.playback', ['examples/recording.jsonl'])]:
            with self.subTest(module=module), tempfile.TemporaryDirectory() as directory:
                endpoint = Path(directory) / 'rpc.json'
                result = subprocess.run([sys.executable, '-u', '-c', code, module, *extra,
                                         '--endpoint', str(endpoint), '--port', '0'],
                                        capture_output=True, text=True, timeout=6)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('provider ready', result.stdout)
                self.assertFalse(endpoint.exists())
