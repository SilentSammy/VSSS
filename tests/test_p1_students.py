import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from vsss import *
from vsss.simulation import VSSSSim


class StudentsTest(unittest.TestCase):
    def test_student_facade_with_local_provider_and_guaranteed_cleanup(self):
        with VSSSSim() as sim:
            with Car(1,provider=sim) as car:
                self.assertTrue(car.is_connected)
                self.assertEqual(car.pose,Pose())
                car.go_to(.1,0,0)
                self.assertEqual(car.read_status().mode_name,'position')
                car.set_velocity(x=.2)
                self.assertEqual(car.read_status().mode_name,'manual')
                self.assertIsNone(car.gps.get_car(999))
            self.assertEqual(sim.cars.status(1).mode_name,'idle')
            self.assertFalse(Car._cars)
            with self.assertRaisesRegex(RuntimeError,'closed'): car.set_velocity(x=.2)

    def test_readonly_car_close_preserves_gate_and_a_newer_owner(self):
        with VSSSSim(wait_for_control=True) as sim:
            with Car(1,provider=sim) as reader:
                reader.read_status()
                reader.restart()
                self.assertTrue(sim._waiting)
            self.assertTrue(sim._waiting)
        with tempfile.TemporaryDirectory() as directory,VSSSSim() as sim:
            endpoint=Path(directory)/'rpc.json'
            with sim.serve(endpoint=endpoint,port=0):
                first=Car(1,endpoint=endpoint)
                with VSSSProvider.connect(endpoint) as other:
                    first.set_velocity(x=.1)
                    other.cars.set_velocity(1,Velocity(.2))
                    first.close()
                    self.assertEqual(sim._velocities[1],Velocity(.2))

    def test_velocity_and_waypoint_scripts_against_separate_simulator(self):
        with tempfile.TemporaryDirectory() as directory, VSSSSim().start() as sim:
            endpoint = Path(directory)/'rpc.json'
            env = dict(os.environ,VSSS_ENDPOINT=str(endpoint))
            with sim.serve(endpoint=endpoint,port=0):
                for script in ['boilerplate_velocity.py','boilerplate_autonomous.py']:
                    with self.subTest(script=script):
                        result = subprocess.run([sys.executable,script],env=env,capture_output=True,text=True,timeout=18)
                        self.assertEqual(result.returncode,0,result.stderr)
                        self.assertEqual(sim.cars.status(1).mode_name,'idle')
                        if script.endswith('autonomous.py'):
                            self.assertIn('Target reached',result.stdout)

    def test_goalie_example_runs_against_seeded_moving_scenario(self):
        from vsss.scenarios import goalie_scenario
        with tempfile.TemporaryDirectory() as directory, VSSSSim(goalie_scenario(123)).start() as sim:
            endpoint=Path(directory)/'rpc.json'
            with sim.serve(endpoint=endpoint,port=0):
                result=subprocess.run([sys.executable,'goalie.py','--duration','1'],
                    env=dict(os.environ,VSSS_ENDPOINT=str(endpoint)),capture_output=True,text=True,timeout=8)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertFalse(sim._waiting)
                self.assertEqual(sim.cars.status(1).mode_name,'idle')

    def test_run_decorator_cleans_up_on_exception(self):
        with VSSSSim() as sim:
            @run
            def program():
                Car(1,provider=sim).set_velocity(x=.2)
                raise ValueError('original problem')
            with self.assertRaisesRegex(ValueError,'original problem'): program()
            self.assertEqual(sim._velocities[1],Velocity())
            self.assertFalse(Car._cars)

    def test_ported_goalie_defends_first_shot_across_five_seeds(self):
        import goalie
        from vsss.scenarios import goalie_scenario
        for seed in (1, 2, 3, 42, 123):
            with self.subTest(seed=seed), VSSSSim(goalie_scenario(seed)) as sim:
                maximum = [-10.]
                def step(dt):
                    sim.step(dt)
                    maximum[0] = max(maximum[0],sim._world.balls[0].pose.y)
                with Car(1,provider=sim) as car, patch.object(goalie,'gps',ClientGPS(provider=sim)), \
                     patch.object(goalie,'monotonic',side_effect=lambda:sim._time), patch.object(goalie,'sleep',side_effect=step):
                    goalie.defend(car,duration=5)
                self.assertLess(maximum[0],1.2)
                self.assertAlmostEqual(sim._world.cars[0].pose.y,1.2,delta=.05)
