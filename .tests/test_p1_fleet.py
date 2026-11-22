from pathlib import Path
import tempfile
from threading import Event as Flag, Thread
import unittest

from vsss import *
from vsss.teleop import LatestCommands


class FleetTest(unittest.TestCase):
    def test_slow_radio_does_not_block_second_car_and_its_stop(self):
        entered, release, second, stopped = Flag(), Flag(), Flag(), Flag()
        class Cars(CarCmd):
            @rpc_method
            def available(self, refresh=False): return [1,2]
            def _apply(self, car_id, value):
                if car_id == 1 and value.x:
                    entered.set(); release.wait(4)
                if car_id == 2:
                    (second if value.x else stopped).set()
        hub = EventHub()
        with tempfile.TemporaryDirectory() as directory, VSSSProvider(GPSState(hub), Cars(hub)) as provider:
            endpoint = Path(directory)/'rpc.json'
            with provider.serve(endpoint=endpoint,port=0), VSSSProvider.connect(endpoint) as client:
                writer = LatestCommands(client.cars, client.fork)
                close_thread = None
                try:
                    writer.submit(1, Velocity(.2))
                    self.assertTrue(entered.wait(2))
                    writer.submit(2, Velocity(.2))
                    self.assertTrue(second.wait(.5), 'car2 blocked behind car1')
                    close_thread = Thread(target=writer.close)
                    close_thread.start()
                    self.assertTrue(stopped.wait(.5), 'car2 STOP blocked behind car1')
                finally:
                    release.set()
                    if close_thread: close_thread.join(5)
                    else: writer.close()
                self.assertFalse(close_thread.is_alive())
                self.assertIsNone(writer.error)
