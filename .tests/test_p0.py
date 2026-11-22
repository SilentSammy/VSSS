from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from vsss import *
from vsss.live import GPSLive
from vsss.playback import GPSPlayback
from vsss.simulation import VSSSSim


class P0Test(unittest.TestCase):
    def test_silent_invalid_missing_and_historical_feedback(self):
        gps = GPSState(EventHub())
        good = WorldState(cars=[CarState(1)], balls=[BallState(0)], captured_at=123)
        gps._update(good)
        self.assertEqual(gps.get_car(1), Pose())
        gps._received -= 1
        self.assertIsNone(gps.get_car(1))
        self.assertFalse(gps.state().valid)
        self.assertEqual(gps.snapshot().cars, good.cars)
        gps._update(replace(good, valid=False))
        self.assertIsNone(gps.get_car(1))
        self.assertEqual(gps.state().balls, [])
        gps._update(replace(good, cars=[]))
        self.assertIsNone(gps.get_car(1))

    def test_live_rejects_delayed_and_reordered_camera_samples(self):
        gps = GPSLive(EventHub(), server=None)
        packet = dict(timestamp=10., frame_sequence=1, frame_age_ms=20, board_detected=True,
                      markers=[dict(id=1, x=.2, y=.3, angle=0)], balls=[])
        try:
            with patch('vsss.live.time.monotonic', return_value=1000.):
                self.assertTrue(gps.ingest(json.dumps(packet)))
                self.assertEqual(gps.get_car(1).x, .2)
            with patch('vsss.live.time.monotonic', return_value=1000.05):
                self.assertFalse(gps.ingest(json.dumps(dict(packet, timestamp=9.9))))
                self.assertEqual(gps.snapshot().sequence, 1)
                self.assertFalse(gps.ingest(json.dumps(dict(packet, timestamp=10.05, frame_age_ms=700))))
            with patch('vsss.live.time.monotonic', return_value=1001.):
                self.assertFalse(gps.ingest(json.dumps(dict(packet, timestamp=10.01))))
            with patch('vsss.live.time.monotonic', return_value=1001.1):
                self.assertTrue(gps.ingest(json.dumps(dict(packet, timestamp=11.1, frame_sequence=2, markers=[]))))
                self.assertIsNone(gps.get_car(1))
        finally: gps.close()

    def test_playback_preserves_source_time_but_uses_fresh_delivery(self):
        gps = GPSPlayback(EventHub())
        gps._receive(WorldState(captured_at=1, frame_age_ms=5000, source_age=5, cars=[CarState(1)]))
        self.assertEqual(gps.snapshot().captured_at, 1)
        self.assertGreater(gps.snapshot().received_at, 1)
        self.assertIsNotNone(gps.get_car(1))
        self.assertLess(gps.age(), .1)

    def test_waypoint_arrival_manual_takeover_stop_and_disconnect(self):
        with VSSSSim() as sim:
            sim.cars.go_to(1, Pose(.2, .2, 90))
            for _ in range(1000):
                sim.step(.01)
                if sim.cars.status(1).mode == 3: break
            status = sim.cars.status(1)
            self.assertEqual(status.mode_name, 'reached')
            self.assertLess(((status.x-.2)**2+(status.y-.2)**2)**.5, .025)
            sim.cars.go_to(1, Pose(-.2, 0))
            sim.cars.set_velocity(1, Velocity(.2))
            self.assertNotIn(1, sim._navigation)
            sim.cars.stop(1)
            self.assertEqual(sim.cars.status(1).mode_name, 'idle')
            sim.cars.go_to(1, Pose(.1, 0))
            sim.cars.disconnect('local')
            self.assertEqual(sim.cars.status(1).mode_name, 'idle')
            self.assertNotIn(1, sim._navigation)

    def test_waypoint_feedback_fault_and_timeout(self):
        with VSSSSim() as sim:
            sim.cars.set_controller(1, dict(telemetry_recovery_ms=100, waypoint_timeout_ms=1000))
            sim.cars.go_to(1, Pose(.6, 1))
            sim.set_feedback(False)
            for _ in range(110): sim.step(.01)
            self.assertEqual(sim.cars.status(1).fault_name, 'telemetry')
            self.assertEqual(sim._velocities[1], Velocity())
        with VSSSSim() as sim:
            sim.cars.set_controller(1, dict(max_speed=.01, min_speed=.001, waypoint_timeout_ms=1000))
            sim.cars.go_to(1, Pose(.6, 1))
            for _ in range(110): sim.step(.01)
            self.assertEqual(sim.cars.status(1).fault_name, 'timeout')

    def test_rpc_waypoint_and_typed_status_use_generic_dispatch(self):
        with tempfile.TemporaryDirectory() as directory, VSSSSim() as sim:
            endpoint = Path(directory) / 'rpc.json'
            with sim.serve(endpoint=endpoint, port=0), VSSSProvider.connect(endpoint) as client:
                command = client.cars.go_to(1, Pose(.1, 0))
                self.assertEqual(command.operation, 'waypoint')
                self.assertIsInstance(command.value, Pose)
                self.assertIsInstance(client.cars.status(1), RobotStatus)
                client.cars.stop(1)
