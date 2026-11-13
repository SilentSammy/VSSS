"""Execute exported code against the real vsss.Car and @run, with fake I/O."""
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from vsss import Car


def verify(source):
    clients=[]
    def factory(**kwargs):
        client=MagicMock()
        client.is_connected=True
        clients.append((kwargs,client))
        return client
    def no_gps(*args,**kwargs):
        raise AssertionError('The velocity-only export must not start GPS')
    with patch('vsss.car.MecanumBLEClient',side_effect=factory), patch('vsss.car.GPS',side_effect=no_gps), \
            patch('vsss.gps.GPS',side_effect=no_gps), patch('time.sleep') as sleep:
        exec(compile(source,'<exported>','exec'),{'__name__':'__main__'})
        assert len(clients)==1,clients
        options,client=clients[0]
        assert options=={'device_name':'car1'},options
        client.set_velocity.assert_called_once_with({'x':.2,'y':0,'w':0})
        client.disconnect.assert_called_once()
        sleep.assert_called_once_with(1)
        assert not Car._cars
        clients.clear()
        try:
            exec(compile(source.replace('sleep(1)','1 / 0'),'<failing-export>','exec'),{'__name__':'__main__'})
        except ZeroDivisionError:
            pass
        else:
            raise AssertionError('Expected the injected program error')
        clients[0][1].disconnect.assert_called_once()
        assert not Car._cars
    print('PASS: real vsss.Car and @run clean up exported code on return and error; GPS stays lazy.')


if __name__=='__main__': verify(Path(sys.argv[1]).read_text(encoding='utf-8-sig'))
