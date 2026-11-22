from dataclasses import replace
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from vsss import CarState, Event, Pose, VSSSProvider, Velocity, WorldState
from vsss.plot import EmbeddedPlot, ProviderFeed, make_plotter, render_state
from vsss.simulation import VSSSSim


def wait_for(predicate):
    deadline = time.monotonic()+4
    while time.monotonic()<deadline:
        if predicate(): return
        time.sleep(.02)
    raise AssertionError('waypoint feed did not update')


class WaypointPlotTest(unittest.TestCase):
    def test_commands_before_and_after_connect_replacement_stop_and_reset(self):
        state = WorldState(cars=[CarState(1,Pose(-.3,0)),CarState(2,Pose(.3,0))])
        with tempfile.TemporaryDirectory() as directory, VSSSSim(state) as sim:
            endpoint = Path(directory)/'rpc.json'
            with sim.serve(endpoint=endpoint,port=0), VSSSProvider.connect(endpoint) as client:
                a,b = Pose(-.3,.5,90),Pose(.3,-.5,-90)
                client.cars.go_to(1,a)
                view = EmbeddedPlot(endpoint,width=240)
                try:
                    wait_for(lambda: view.feed.waypoints()=={1:a})
                    client.cars.go_to(2,b)
                    wait_for(lambda: view.feed.waypoints()=={1:a,2:b})
                    frame,_ = view.frame()
                    plain = render_state(view.plotter,view.feed.snapshot()[0])
                    self.assertFalse(np.array_equal(frame,plain))
                    replacement = Pose(-.2,.6,45)
                    client.cars.go_to(1,replacement)
                    wait_for(lambda: view.feed.waypoints().get(1)==replacement)
                    client.cars.set_velocity(1,Velocity(.1))
                    wait_for(lambda: view.feed.waypoints()=={2:b})
                    client.cars.stop(2)
                    wait_for(lambda: not view.feed.waypoints())
                    client.cars.go_to(1,a)
                    wait_for(lambda: view.feed.waypoints()=={1:a})
                    client.load_scenario(state)
                    wait_for(lambda: not view.feed.waypoints())
                    client.cars.go_to(1,a)
                    wait_for(lambda: view.feed.waypoints()=={1:a})
                    # Fresh camera feedback at the target hides a reached marker.
                    view.feed._receive(Event('gps',replace(state,cars=[CarState(1,a)]),1,0))
                    self.assertEqual(view.feed.waypoints(),{})
                    self.assertEqual(sim.cars.observed()[0].operation,'waypoint')
                finally:
                    view.close()
                # The plot observes commands; closing it must not cancel control.
                self.assertEqual(sim.cars.observed()[0].operation,'waypoint')

    def test_marker_draws_at_field_coordinates_and_shows_heading(self):
        plotter = make_plotter(width=360)
        state = WorldState(cars=[CarState(1,Pose(-.3,-.5))])
        target = Pose(.25,.5,90)
        plain = render_state(plotter,state)
        marked = render_state(plotter,state,{1:target})
        x=round(plotter.width/2+target.x*plotter.scale)
        y=round(plotter.height/2-target.y*plotter.scale)
        self.assertTrue(np.any(marked[y-15:y+15,x-15:x+15] != plain[y-15:y+15,x-15:x+15]))
        self.assertEqual(marked.shape,plain.shape)
