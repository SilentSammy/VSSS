"""Discovery caching and teleop assignment without hardware or input hooks."""
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import AsyncMock, Mock, patch

from tools.vsss_teleop import assign_inputs, explicit_assignments, velocity, main
from vsss._provider import Provider


class TeleopTests(unittest.TestCase):
    def test_teleop_sends_only_rounded_input_changes(self):
        import vsss
        inp = SimpleNamespace(KEYBOARD_AVAILABLE=True, pressed_keys=set())
        client = Mock(is_connected=True, resolution=.05, connection_generation=1)
        commands = [dict(x=x, y=0, w=0) for x in (0, .01, .1, .11)]
        with patch.dict(sys.modules, {'vsss.input': inp, 'inputs': SimpleNamespace(
                devices=SimpleNamespace(gamepads=[]))}), \
             patch.object(vsss, 'input', inp, create=True), \
             patch('tools.vsss_teleop.MecanumBLEClient', return_value=client), \
             patch('tools.vsss_teleop.velocity', side_effect=commands), \
             patch('tools.vsss_teleop.time.sleep', side_effect=[None, None, None, KeyboardInterrupt]), \
             patch('builtins.print'):
            main(['--car', '1=wasd'])
        self.assertEqual(client.set_velocity.call_count, 2)
        self.assertEqual(client.set_velocity.call_args_list[0].args[0]['x'], 0)
        self.assertEqual(client.set_velocity.call_args_list[1].args[0]['x'], .1)
        client.disconnect.assert_called_once()

    def test_explicit_assignments_support_combined_and_default_inputs(self):
        self.assertEqual(explicit_assignments(['3=wasd+gamepad:0', '8'],
            ['wasd', 'ijkl', 'gamepad:0']),
            [(3, ('wasd', 'gamepad:0')), (8, ('ijkl',))])

    def test_invalid_explicit_assignments_are_rejected(self):
        for specs in (['1', '1'], ['1=wasd', '2=wasd'], ['1=gamepad:9'],
                      ['-1'], ['1='], ['1=wasd+wasd']):
            with self.subTest(specs=specs), self.assertRaises(ValueError):
                explicit_assignments(specs, ['wasd', 'ijkl', 'gamepad:0'])

    def test_list_exits_without_loading_input_or_connecting(self):
        with patch('tools.vsss_teleop.available_cars', return_value=[2, 5]) as scan, \
             patch('tools.vsss_teleop.MecanumBLEClient') as client, \
             patch('builtins.print') as output:
            main(['--list', '--refresh'])
            scan.assert_called_once_with(refresh=True)
            client.assert_not_called()
            output.assert_called_once_with([2, 5])

    def test_single_car_uses_keyboard_and_first_pad(self):
        self.assertEqual(assign_inputs([4], ['wasd', 'ijkl', 'gamepad:0']),
                         [(4, ('wasd', 'gamepad:0'))])

    def test_assignments_consume_inputs_and_allow_early_finish(self):
        answers = iter(['99', '2', '4', '1', '1', '2', ''])
        with patch('builtins.print'):
            result = assign_inputs([1, 2, 3], ['wasd', 'ijkl', 'gamepad:0'], lambda _: next(answers))
        self.assertEqual(result, [(2, ('wasd',)), (1, ('gamepad:0',))])

    def test_keyboard_assignment_does_not_read_gamepad(self):
        inp = Mock()
        inp.get_bipolar_ctrl.side_effect = [0, 1, 0, -1]
        self.assertEqual(velocity(('wasd',), inp), dict(x=.5, y=0, w=.5))
        inp.get_axis.assert_not_called()

    def test_discovery_filters_names_caches_and_includes_connected_cars(self):
        provider = Provider()
        devices = {name: (SimpleNamespace(name=name), SimpleNamespace(local_name=name))
                   for name in ('car1', 'car7', 'carpet', 'other')}
        scan = AsyncMock(return_value=devices)
        with patch('bleak.BleakScanner.discover', scan):
            self.assertEqual(provider.available_cars(), [1, 7])
            provider.controllers['car1'] = None
            self.assertEqual(provider.available_cars(), [1, 7])
            self.assertEqual(scan.call_count, 1)
            self.assertEqual(provider.available_cars(refresh=True), [1, 7])
            self.assertEqual(scan.call_count, 2)


if __name__ == '__main__':
    unittest.main()
