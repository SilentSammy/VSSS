"""Real UDP reception and student-facing composition without a physical robot."""
import json
import socket
import struct
import time
import unittest
from unittest.mock import patch

from car import Car
from gps import GPS, Pose, Position
from mp.axis_trim import AxisTrim


class GPSTests(unittest.TestCase):
    def setUp(self):
        self.gps = GPS(port=0, stale_after=.15)
        self.sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def tearDown(self):
        self.sender.close()
        self.gps.close()

    def send(self, packet):
        self.sender.sendto(json.dumps(packet).encode(), ('127.0.0.1', self.gps.port))

    def wait_for(self, predicate):
        deadline = time.monotonic() + 1
        while not predicate():
            if time.monotonic() > deadline:
                self.fail('Receiver did not update')
            time.sleep(.005)

    def packet(self, markers=None, balls=None, board=True):
        return dict(board_detected=board,
                    markers=markers if markers is not None else [dict(id=4, x=.2, y=-.1, angle=90)],
                    balls=balls if balls is not None else [dict(x=.3, y=.4)])

    def test_singleton_and_live_objects(self):
        self.assertIs(GPS(), self.gps)
        self.assertEqual(self.gps.cars, {})
        with self.assertRaises(ValueError):
            GPS(port=self.gps.port + 1)
        self.send(self.packet())
        self.wait_for(lambda: self.gps.get_car(4) is not None)
        self.assertEqual(self.gps.get_car(4), Pose(.2, -.1, 90))
        self.assertEqual(self.gps.balls, (Position(.3, .4),))
        self.gps.cars.clear()
        self.assertIn(4, self.gps.cars)

    def test_te2004b_packets_without_balls_work_in_passive_mode(self):
        self.gps.close()
        self.gps=GPS(port=0,subscribe=False)
        packet=self.packet()
        del packet['balls']
        self.send(packet)
        self.wait_for(lambda:self.gps.get_car(4) is not None)
        self.assertEqual(self.gps.balls,())
        self.assertIsNone(self.gps._server)
        self.assertFalse(self.gps.subscribe)
        self.assertIs(GPS(),self.gps)
        with self.assertRaises(ValueError):
            GPS(subscribe=True)

    def test_delayed_backlog_does_not_refresh_pose_and_fresh_packet_recovers(self):
        received=[]
        self.gps.add_listener(received.append)
        packet=self.packet();packet.update(timestamp=100.,frame_age_ms=0.)
        self.send(packet)
        self.wait_for(lambda:len(received)==1)
        self.assertTrue(received[-1].usable)
        first_receipt=received[-1].received_at
        time.sleep(.22)
        packet['timestamp']=100.01
        self.send(packet)
        self.wait_for(lambda:len(received)==2)
        self.assertFalse(received[-1].usable)
        self.assertGreater(received[-1].network_excess_delay_s,.15)
        self.assertIsNone(self.gps.get_car(4))
        self.assertIn('delayed',self.gps.feedback_problem)
        packet['timestamp']=100.+time.monotonic()-first_receipt+.02
        self.send(packet)
        self.wait_for(lambda:len(received)==3)
        self.assertTrue(received[-1].usable)
        self.assertIsNotNone(self.gps.get_car(4))

    def test_frame_old_on_server_is_recorded_but_not_used(self):
        received=[]
        self.gps.add_listener(received.append)
        packet=self.packet();packet.update(timestamp=100.,frame_age_ms=200.)
        self.send(packet)
        self.wait_for(lambda:len(received)==1)
        self.assertFalse(received[0].usable)
        self.assertIsNone(self.gps.get_car(4))
        packet.update(timestamp=100.01,frame_age_ms=10.)
        self.send(packet)
        self.wait_for(lambda:len(received)==2)
        self.assertTrue(received[1].usable)
        self.assertIsNotNone(self.gps.get_car(4))

    def test_frame_age_contributes_to_expiry_after_receipt(self):
        self.gps.stale_after=.5
        packet=self.packet();packet.update(timestamp=100.,frame_age_ms=350.)
        self.send(packet)
        self.wait_for(lambda:self.gps.get_car(4) is not None)
        time.sleep(.2)
        self.assertIsNone(self.gps.get_car(4))

    def test_out_of_order_frame_is_logged_but_does_not_replace_pose(self):
        self.gps.stale_after=1.
        received=[]
        self.gps.add_listener(received.append)
        packet=self.packet();packet.update(timestamp=100.1,frame_age_ms=0.)
        self.send(packet)
        self.wait_for(lambda:len(received)==1)
        packet['timestamp']=100.0
        packet['markers'][0]['x']=9.
        self.send(packet)
        self.wait_for(lambda:len(received)==2)
        self.assertFalse(received[-1].usable)
        self.assertEqual(self.gps.get_car(4),Pose(.2,-.1,90))

    def test_missing_and_stale_targets(self):
        self.send(self.packet())
        self.wait_for(lambda: self.gps.board_detected)
        self.send(self.packet(markers=[]))
        self.wait_for(lambda: self.gps.get_car(4) is None)
        self.assertTrue(self.gps.balls)
        self.send(self.packet(board=False))
        self.wait_for(lambda: not self.gps.board_detected)
        self.assertFalse(self.gps.balls)
        self.send(self.packet())
        self.wait_for(lambda: self.gps.board_detected)
        time.sleep(.2)
        self.assertIsNone(self.gps.get_car(4))
        self.assertFalse(self.gps.balls)
        self.assertFalse(self.gps.board_detected)

    def test_invalid_packets_do_not_kill_receiver_or_renew_freshness(self):
        self.send(self.packet())
        self.wait_for(lambda: self.gps.board_detected)
        for packet in ([], {}, self.packet(balls=[dict(x=float('nan'), y=0)]),
                       self.packet(markers=[dict(id=4)])):
            self.send(packet)
        time.sleep(.2)
        self.assertFalse(self.gps.board_detected)
        self.send(self.packet())
        self.wait_for(lambda: self.gps.board_detected)

    def test_car_uses_same_gps_and_number_and_cleans_up(self):
        self.send(self.packet())
        self.wait_for(lambda: self.gps.board_detected)
        with patch('car.MecanumBLEClient') as client_type:
            client = client_type.return_value
            client.is_connected = True
            with Car(4) as car:
                client_type.assert_called_once_with(device_name='car4')
                client.set_trim.assert_not_called()
                self.assertIs(car.gps, self.gps)
                self.assertEqual(car.position, (.2, -.1))
                self.assertEqual(car.pose.angle, 90)
                car.set_velocity(.5, -.2, .1)
                client.set_velocity.assert_called_with(dict(x=.5, y=-.2, w=.1))
                matrix=((.9,0.,0.),(.02,1.1,0.),(-.04,0.,1.))
                car.set_trim(matrix)
                client.set_trim.assert_called_with(matrix)
                with self.assertRaises(ValueError):
                    car.set_velocity(float('nan'))
                client.is_connected = False
                with self.assertRaises(ConnectionError):
                    car.set_velocity(.5)
            client.disconnect.assert_called_once()

    def test_car_applies_saved_calibration_transparently(self):
        with patch('car.MecanumBLEClient') as client_type:
            client_type.return_value.is_connected = True
            with Car(1):
                matrix = client_type.return_value.set_trim.call_args.args[0]
                self.assertEqual(len(matrix), 3)
                self.assertAlmostEqual(matrix[2][0], -0.05721879625165137)

    def test_failed_connection_is_closed(self):
        with patch('car.MecanumBLEClient') as client_type:
            client_type.return_value.wait_until_connected.side_effect = TimeoutError
            with self.assertRaises(TimeoutError):
                Car(4)
            client_type.return_value.disconnect.assert_called_once()

    def test_packet_observer_preserves_timing_and_cannot_change_shared_poses(self):
        received=[]
        self.gps.add_listener(received.append)
        packet=self.packet()
        packet.update(timestamp=123., frame_sequence=42, frame_age_ms=87.)
        self.send(packet)
        self.wait_for(lambda: bool(received))
        sample=received[0]
        self.assertEqual(sample.frame_sequence,42)
        self.assertEqual(sample.frame_age_ms,87.)
        self.assertEqual(sample.server_timestamp,123.)
        self.assertEqual(sample.cars[4],Pose(.2,-.1,90))
        self.assertGreater(sample.received_at,0)
        with self.assertRaises(TypeError):
            sample.cars[4]=Pose(0,0,0)
        self.gps.remove_listener(received.append)

    def test_position_diagnostics_are_reported_once_and_recover(self):
        with patch('car.MecanumBLEClient'), patch('builtins.print') as output:
            with Car(4) as car:
                self.assertIsNone(car.pose)
                self.assertIsNone(car.pose)
                self.assertEqual(output.call_count, 1)
                self.assertIn('No camera telemetry', output.call_args.args[0])
                self.send(self.packet())
                self.wait_for(lambda: self.gps.board_detected)
                self.assertEqual(car.wait_for_position(.2), Pose(.2, -.1, 90))
                self.assertIn('Camera position available', output.call_args.args[0])
                output.reset_mock()
                self.send(self.packet(markers=[]))
                self.wait_for(lambda: self.gps.get_car(4) is None)
                self.assertIsNone(car.pose)
                self.assertIn('Marker 4', output.call_args.args[0])

    def test_wait_for_position_timeout_cleans_up_connection(self):
        with patch('car.MecanumBLEClient') as client_type:
            with self.assertRaisesRegex(TimeoutError, 'No camera telemetry'):
                with Car(4) as car:
                    car.wait_for_position(.01)
            client_type.return_value.disconnect.assert_called_once()


class TrimTests(unittest.TestCase):
    def test_cross_axis_terms_cancel_a_known_linear_drift(self):
        trim = AxisTrim()
        payload = struct.pack('<9f', 1.,0.,0., -.1,1.,0., -.2,0.,1.)
        self.assertEqual(trim.write(payload), payload)
        for requested in ((.5,0.,0.), (-.5,0.,0.), (.6,.2,.1)):
            x,y,w = trim.apply(requested)
            actual = (x, y+.1*x, w+.2*x)
            for measured,target in zip(actual,requested):
                self.assertAlmostEqual(measured,target,places=6)
        self.assertEqual(trim.apply((0.,0.,0.)), (0.,0.,0.))

    def test_full_matrix_mapping_validation_and_diagonal_matrix(self):
        trim = AxisTrim()
        values = (1., .1, -.2, -.1, 1., .2, -.04, .03, 1.)
        payload = struct.pack('<9f', *values)
        self.assertEqual(trim.write(payload), payload)
        for actual,expected in zip(trim.apply((.5,-.3,.2)),(.43,-.31,.171)):
            self.assertAlmostEqual(actual,expected,places=6)
        for bad in (b'', bytes(12), bytes(35), bytes(37)):
            self.assertEqual(trim.write(bad),payload)
        for index,value in ((0,-.1),(4,2.1),(1,-2.1),(7,2.1),(5,float('nan')),(8,float('inf'))):
            invalid=list(values);invalid[index]=value
            self.assertEqual(trim.write(struct.pack('<9f',*invalid)),payload)
        trim.write(struct.pack('<9f',.5,0.,0.,0.,1.5,0.,0.,0.,2.))
        self.assertEqual(struct.unpack('<9f',trim.read()),(.5,0.,0.,0.,1.5,0.,0.,0.,2.))
        self.assertEqual(trim.apply((1.,-1.,.25)),(.5,-1.,.5))

    def test_validated_trim_keeps_zero_and_clamps_commands(self):
        trim = AxisTrim()
        self.assertEqual(trim.apply((0, 0, 0)), (0, 0, 0))
        payload = struct.pack('<9f', .5,0.,0.,0.,1.5,0.,0.,0.,2.)
        self.assertEqual(trim.write(payload), payload)
        self.assertEqual(trim.apply((1., -1., .25)), (.5, -1., .5))
        for invalid in (b'bad', struct.pack('<fff', -1., 1., 1.),
                        struct.pack('<fff', float('nan'), 1., 1.),
                        struct.pack('<fff', float('inf'), 1., 1.)):
            self.assertEqual(trim.write(invalid), payload)
        self.assertEqual(trim.apply((0, 0, 0)), (0, 0, 0))
