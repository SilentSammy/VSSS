"""Verify the example lines up before requesting full forward speed."""
import builtins
from pathlib import Path
from types import SimpleNamespace
import unittest

from vsss import Pose, Velocity
from vsss.scenarios import load_scenario
from vsss.simulation import VSSSSim


class KickTest(unittest.TestCase):
    def test_waypoint_alignment_precedes_full_speed_strike(self):
        root = Path(__file__).resolve().parent
        code = (root/'examples/08_kick_top_wall.py').read_text()
        with VSSSSim(load_scenario(root.parent/'examples/car1_ball1.jsonl')) as sim:
            target = Pose(.3,0,90)
            strikes = []
            test = self
            class Car:
                def __init__(self, number): self.number=number
                def wait_for_position(self): return sim._world.cars[0].pose
                def go_to(self, x, y, angle):
                    test.assertAlmostEqual(x,target.x)
                    test.assertAlmostEqual(y,target.y)
                    test.assertEqual(angle,target.angle)
                    sim.cars.go_to(self.number,Pose(x,y,angle))
                def read_status(self): return sim.cars.status(self.number)
                def set_velocity(self, **kwargs):
                    pose=sim._world.cars[0].pose
                    test.assertAlmostEqual(pose.x,target.x,delta=.03)
                    test.assertAlmostEqual(pose.y,target.y,delta=.03)
                    test.assertAlmostEqual(pose.angle,target.angle,delta=5)
                    test.assertEqual(kwargs,dict(x=1,y=0,w=0))
                    strikes.append(pose)
                    sim.cars.set_velocity(self.number,Velocity(**kwargs))
                def stop(self): sim.cars.stop(self.number)
            def sleep(dt):
                for _ in range(round(dt/.01)): sim.step(.01)
            def imports(name,*args,**kwargs):
                if name=='vsss': return SimpleNamespace(Car=Car,gps=SimpleNamespace(ball=sim._world.balls[0].pose),run=lambda fn:fn)
                if name=='time': return SimpleNamespace(sleep=sleep,monotonic=lambda:sim._time)
                return builtins.__import__(name,*args,**kwargs)
            scope={'__name__':'example','__builtins__':dict(vars(builtins),__import__=imports)}
            exec(code,scope)
            scope['main']()
            self.assertEqual(len(strikes),1)
            self.assertGreater(sim._world.balls[0].pose.y,.4)
            self.assertEqual(sim.cars.observed()[0].value,Velocity())
