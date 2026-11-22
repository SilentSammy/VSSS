from collections import deque
from pathlib import Path
import tempfile
from threading import Lock
import time
from types import SimpleNamespace
import unittest

import cv2

from vsss import *
from vsss.plot import ProviderFeed, SimDrag
from vsss.simulation import VSSSSim


class EditingTest(unittest.TestCase):
    def test_exclusive_grab_anchor_clamp_release_and_eof(self):
        with tempfile.TemporaryDirectory() as directory, VSSSSim() as sim:
            endpoint = Path(directory)/'rpc.json'
            with sim.serve(endpoint=endpoint, port=0), VSSSProvider.connect(endpoint) as a, VSSSProvider.connect(endpoint) as b:
                source = a.describe()['source_id']
                a.cars.go_to(1, Pose(.2,0))
                grab = a.edit('grab','car',1,source_id=source)
                self.assertNotIn(1, sim._navigation)
                with self.assertRaises(RuntimeError): b.edit('grab','car',1,source_id=source)
                with self.assertRaises(RuntimeError): b.cars.set_velocity(1,Velocity(.2))
                with self.assertRaises(RuntimeError): b.update_state(StatePatch(cars=[CarState(1,Pose(.1,0))]))
                a.edit('move','car',1,Pose(3,4,45),source_id=source,token=grab['token'])
                before = sim._world.cars[0].pose
                for _ in range(100): sim.step(.01)
                self.assertEqual(sim._world.cars[0].pose, before)
                self.assertLess(before.x, .75)
                self.assertLess(before.y, 1.45)
                a.edit('release','car',1,source_id=source,token=grab['token'])
                self.assertEqual(sim._physics.cars[1].velocity.length, 0)
                ball = a.edit('grab','ball',0,source_id=source)
                a.close()
                deadline = time.monotonic()+2
                while sim._held and time.monotonic()<deadline: time.sleep(.01)
                self.assertEqual(sim._held,{})

    def test_scenario_reset_invalidates_tokens_and_foreign_source_is_rejected(self):
        with VSSSSim() as sim:
            grab = sim.edit('grab','car',1,source_id=sim.source_id)
            state = sim.scenario().state
            sim.load_scenario(state)
            with self.assertRaises(ValueError):
                sim.edit('move','car',1,Pose(.2,0),source_id=sim.source_id,token=grab['token'])
            with self.assertRaises(ValueError): sim.edit('grab','car',1,source_id='old-provider')
            self.assertEqual(sim._held,{})

    def test_mouse_move_rotate_ball_and_release_queue_ordering(self):
        world = WorldState(cars=[CarState(1)],balls=[BallState(0,Pose(.3,0))])
        calls = []
        info = dict(capabilities=dict(simulation_edit=True),simulation=dict(
            body=[[-.06,-.06],[.06,-.06],[.06,.06],[-.06,.06]],ball_radius=.02,generation=0))
        feed = SimpleNamespace(snapshot=lambda:(world,'simulation','source'),info=lambda:info,
                               gesture=lambda *args,**kwargs:calls.append((args,kwargs)))
        drag = SimDrag(SimpleNamespace(width=200,height=400,scale=100),feed)
        drag.mouse(cv2.EVENT_LBUTTONDOWN,100,200,0)
        drag.mouse(cv2.EVENT_LBUTTONUP,110,190,0)
        self.assertEqual([call[0][0] for call in calls],['grab','move','release'])
        self.assertEqual(calls[1][0][3],Pose(.1,.1))
        calls.clear()
        drag.mouse(cv2.EVENT_LBUTTONDOWN,105,200,0)
        drag.mouse(cv2.EVENT_LBUTTONUP,100,195,0)
        self.assertAlmostEqual(calls[1][0][3].angle,90)
        calls.clear()
        drag.mouse(cv2.EVENT_LBUTTONDOWN,130,200,0)
        drag.mouse(cv2.EVENT_LBUTTONUP,140,200,0)
        self.assertEqual(calls[0][0][1:3],('ball',0))
        # The worker's queue coalesces moves without dropping release boundaries.
        real = ProviderFeed.__new__(ProviderFeed)
        real.lock,real._edits,real.source_id,real.description=Lock(),deque(),'source',info
        real.gesture('grab','car',1,source_id='source')
        for x in (.1,.2,.3): real.gesture('move','car',1,Pose(x,0),source_id='source')
        real.gesture('release','car',1,source_id='source')
        self.assertEqual([item[0] for item in real._edits],['grab','move','release'])
        self.assertEqual(real._edits[1][3],Pose(.3,0))
