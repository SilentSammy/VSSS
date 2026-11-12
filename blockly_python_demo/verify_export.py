"""Check exported Blockly source against real car.py with fake I/O only."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def verify(source):
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    client = MagicMock()
    client.is_connected = True
    client.read_status.return_value = None
    transport = SimpleNamespace(MecanumBLEClient=MagicMock(return_value=client))
    # Install fake transport before loading car.py, so even its imports cannot
    # load the BLE/UI modules. GPS's class is imported but never instantiated.
    with patch.dict(sys.modules, {'mecanum_client': transport}):
        spec = importlib.util.spec_from_file_location('contract_car', root / 'car.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    gps = MagicMock()
    gps.get_car.return_value = SimpleNamespace(x=0., y=0., angle=90.)
    with patch.object(module, 'GPS', return_value=gps), \
            patch.dict(sys.modules, {'car': module}), patch('time.sleep') as sleep:
        exec(compile(source, '<exported-blocks>', 'exec'), {})
        transport.MecanumBLEClient.assert_called_once_with(device_name='car1')
        client.set_velocity.assert_called_once_with({'x': .2, 'y': 0, 'w': 0})
        client.set_waypoint.assert_called_once_with(0, .5, 90)
        client.stop.assert_called()
        client.disconnect.assert_called_once()
        assert [call.args[0] for call in sleep.call_args_list] == [1, 2]
        # The same generated context manager must disconnect on a runtime error.
        client.reset_mock()
        try:
            exec(compile(source.replace('sleep(1)', '1 / 0'), '<failing-export>', 'exec'), {})
        except ZeroDivisionError:
            pass
        else:
            raise AssertionError('Expected the injected program error')
        client.disconnect.assert_called_once()
    print('PASS: exported source uses real Car API and disconnects on success and error (all I/O mocked).')


if __name__ == '__main__':
    verify(Path(sys.argv[1]).read_text(encoding='utf-8'))
