import json
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys
import tempfile
from threading import Thread
import time
from types import SimpleNamespace
import unittest

import pymunk

from vsss import *
from vsss.physics import Physics
from vsss.playback import VSSSPlayback
from vsss.recording import Recorder
from vsss.scenarios import load_scenario
from vsss.simulation import VSSSSim


class AcceptanceTest(unittest.TestCase):
    def test_old_normal_impact_gain_semantics_and_rolling_distance(self):
        world=WorldState(cars=[CarState(1,Pose(-.2,0))],balls=[BallState(0)])
        physics=Physics(world,dict(ball_model=dict(impact_gain=.88)))
        car,ball=next(iter(physics.cars[1].shapes)),next(iter(physics.balls[0].shapes))
        car.body.velocity=(.5,0)
        arbiter=SimpleNamespace(shapes=(car,ball),normal=pymunk.Vec2d(1,0),
            contact_point_set=SimpleNamespace(points=[SimpleNamespace(point_a=(0,0))]))
        physics._ball_contact(arbiter,None,None)
        ball.body.velocity=(.9,.2)  # Solver's tangential response must be retained.
        physics._ball_impact(arbiter,None,None)
        self.assertAlmostEqual(ball.body.velocity.x,.44)
        self.assertAlmostEqual(ball.body.velocity.y,.2)
        state=WorldState(field=Field(6,6),balls=[BallState(0)])
        with VSSSSim(SimScenario(state,[BodyMotion('ball',0,.4,0)],
                    dict(ball_model=dict(rolling_deceleration=.1,stop_speed=0,
                                         drift_acceleration_x=0,drift_acceleration_y=0),
                         ball_wobble=dict(enabled=False)))) as sim:
            for _ in range(500): sim.step(.01)
            self.assertAlmostEqual(sim._world.balls[0].pose.x,.4**2/(2*.1),delta=.003)

    def test_simulator_cli_exports_seeded_scenario_and_hot_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            endpoint,scenario=Path(directory)/'rpc.json',Path(directory)/'shot.jsonl'
            process=subprocess.Popen([sys.executable,'-u','-m','vsss.simulation','--scenario','goalie',
                '--seed','123','--save-scenario',str(scenario),'--config','examples/physics_lab.json',
                '--endpoint',str(endpoint),'--port','0'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            output=Queue()
            Thread(target=lambda:[output.put(line.strip()) for line in process.stdout],daemon=True).start()
            try:
                while 'provider ready' not in output.get(timeout=8): pass
                saved=load_scenario(scenario)
                self.assertEqual(saved.seed,123)
                self.assertTrue(saved.wait_for_control)
                with VSSSProvider.connect(endpoint) as client:
                    identity=client.describe()['source_id']
                    result=subprocess.run([sys.executable,'-m','vsss.simulation','--load',str(scenario),
                        '--endpoint',str(endpoint)],capture_output=True,text=True,timeout=5)
                    self.assertEqual(result.returncode,0,result.stderr)
                    self.assertEqual(client.describe()['source_id'],identity)
                    self.assertTrue(client.describe()['simulation']['waiting'])
            finally:
                process.terminate(); process.communicate(timeout=5)

    def test_record_and_replay_waypoint_and_historical_onboard_status(self):
        with tempfile.TemporaryDirectory() as directory, VSSSSim() as sim:
            endpoint,recording=Path(directory)/'rpc.json',Path(directory)/'recording.jsonl'
            with sim.serve(endpoint=endpoint,port=0),VSSSProvider.connect(endpoint) as client:
                client.cars.status(1)
                with Recorder(client,recording):
                    client.cars.go_to(1,Pose(.1,0))
                    expected=client.cars.status(1)
                    deadline=time.monotonic()+2
                    while 'car_status' not in recording.read_text() and time.monotonic()<deadline: time.sleep(.01)
            with VSSSPlayback(recording,speed=100,wait_for_subscriber=False).start() as replay:
                self.assertTrue(replay.finished.wait(3))
                self.assertIsNone(replay.error)
                self.assertEqual(replay.cars.status(1),expected)
                self.assertEqual(replay.cars.observed()[0].operation,'waypoint')
                with self.assertRaises(RuntimeError): replay.cars.go_to(1,Pose(.2,0))
