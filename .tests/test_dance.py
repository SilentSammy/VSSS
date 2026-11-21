"""Fleet discovery and motion lifecycle without hardware."""
import unittest
from unittest.mock import Mock, patch

from tools import vsss_dance


class DanceTests(unittest.TestCase):
    def run_dance(self, argv, discovery=False, interrupt=False):
        clock = [0.]
        clients = []
        events = []

        def client(**kwargs):
            result = Mock(is_connected=True, connection_generation=1)
            clients.append(result)
            return result

        def scanner(**kwargs):
            queue = kwargs['args'][0]
            events.append(queue)
            return Mock(start=lambda: queue.put(('cars', [1])))

        def sleep(seconds):
            clock[0] += .5
            if discovery and len(clients) == 1:
                events[0].put(('cars', [1, 2]))
            if interrupt and clock[0] >= 1.5:
                raise KeyboardInterrupt

        with patch.object(vsss_dance, 'MecanumBLEClient', side_effect=client), \
                patch.object(vsss_dance, 'Thread', side_effect=scanner), \
                patch.object(vsss_dance.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(vsss_dance.time, 'sleep', side_effect=sleep), patch('builtins.print'):
            vsss_dance.main(argv)
        return clients

    def test_alternates_and_stops_on_timed_exit(self):
        clients = self.run_dance(['--car', '1', '--duration', '2.5'])
        self.assertEqual([call.args[0]['w'] for call in clients[0].set_velocity.call_args_list],
                         [.2, -.2, .2, 0.])
        clients[0].stop.assert_called_once()
        clients[0].disconnect.assert_called_once()

    def test_new_cars_join_without_reconnecting_existing_car(self):
        clients = self.run_dance(['--duration', '2.5'], discovery=True)
        self.assertEqual(len(clients), 2)
        for client in clients:
            client.connect.assert_called_once()
            client.stop.assert_called_once()
            client.disconnect.assert_called_once()

    def test_interrupt_stops_every_car(self):
        clients = self.run_dance(['--car', '1', '--car', '2'], interrupt=True)
        for client in clients:
            self.assertEqual(client.set_velocity.call_args.args[0]['w'], 0.)
            client.stop.assert_called_once()
            client.disconnect.assert_called_once()


if __name__ == '__main__':
    unittest.main()
