import ast
import math
from pathlib import Path
import struct
from types import SimpleNamespace
import unittest

from mp.onboard_controller import OnboardController


class WaypointStreamTest(unittest.TestCase):
    def test_retarget_retains_measured_motion_but_stop_resets(self):
        controller=OnboardController()
        controller.set_target(1,0,0)
        controller.update((0,0,0),100)
        controller.update((.01,0,0),150)
        previous,velocity=controller.previous,controller.velocity[:]
        self.assertGreater(velocity[0],0)
        controller.set_target(.9,0,0,preserve_motion=True)
        self.assertEqual(controller.previous,previous)
        self.assertEqual(controller.velocity,velocity)
        controller.clear_target()
        self.assertIsNone(controller.previous)
        self.assertEqual(controller.velocity,[0,0,0])

    def test_duplicate_waypoint_does_not_reset_arrival_or_tracking(self):
        source=Path(__file__).resolve().parents[1]/'mp/main.py'
        tree=ast.parse(source.read_text())
        handler=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='on_waypoint')
        controller=OnboardController()
        controller.set_target(.2,.3,math.pi/2)
        controller.update((.2,.3,90),100)
        state=dict(reconnect_at=None,restart_at=None,waypoint=(.2,.3,90),mode=2,waypoint_id=1)
        scope=dict(state=state,controller=controller,struct=struct,math=math,MODE_POSITION=2,
                   FAULT_NONE=0,freshness=SimpleNamespace(reset=lambda:None))
        exec(compile(ast.Module(body=[handler],type_ignores=[]),str(source),'exec'),scope)
        # Compare after packing: the wire uses single-precision coordinates.
        packet=struct.pack('<fff',.2,.3,90)
        state['waypoint']=struct.unpack('<fff',packet)
        previous=controller.previous
        controller.stable_since=100
        self.assertEqual(scope['on_waypoint'](packet),packet)
        self.assertEqual(controller.previous,previous)
        self.assertEqual(controller.stable_since,100)
        self.assertEqual(state['waypoint_id'],1)
        scope['on_waypoint'](struct.pack('<fff',.25,.3,90))
        self.assertIsNone(controller.target)
        self.assertEqual(controller.previous,previous)
        self.assertEqual(state['waypoint_id'],2)
