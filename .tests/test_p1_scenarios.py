from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

from vsss import *
from vsss.scenarios import goalie_scenario, load_scenario, physical_config, printable_field, save_scenario
from vsss.simulation import VSSSSim


class ScenarioTest(unittest.TestCase):
    def test_default_simulator_and_generated_scenarios_use_printable_field(self):
        from aruco_gps.vision_setup import DEFAULT_SETUP
        field = Field(*DEFAULT_SETUP.board.get_print_dimensions())
        self.assertEqual(printable_field(), field)
        self.assertEqual(goalie_scenario().state.field, field)
        with VSSSSim() as sim:
            self.assertEqual(sim.gps.state().field, field)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'frames.jsonl'
            path.write_text('\n'.join(json.dumps(value) for value in [
                dict(type='sim_scenario', schema=1),
                dict(time=1, cars=[], balls=[]), dict(time=2, cars=[], balls=[]),
            ]))
            self.assertEqual(load_scenario(path).state.field, field)

    def test_goalie_seed_export_wait_gate_and_control_release(self):
        scenario = goalie_scenario(123)
        self.assertEqual(goalie_scenario(123), scenario)
        self.assertNotEqual(goalie_scenario(124), scenario)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'shot.jsonl'
            save_scenario(scenario, path)
            self.assertEqual(load_scenario(path), scenario)
            with self.assertRaises(FileExistsError): save_scenario(scenario, path)
        with VSSSSim(scenario) as sim:
            initial = sim._world.balls[0].pose
            sim.cars.connect(1)
            sim.cars.status(1)
            for _ in range(100): sim.step(.01)
            self.assertEqual(sim._world.balls[0].pose, initial)
            sim.cars.set_velocity(1, Velocity())
            sim.step(.1)
            self.assertGreater(sim._world.balls[0].pose.y, initial.y)

    def test_two_frames_reconstruct_world_motion_and_angle_wrap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'frames.jsonl'
            a = dict(time=1, cars=[dict(id=1,x=0,y=0,angle=179)], balls=[dict(x=.3,y=0)])
            b = dict(time=1.1, cars=[dict(id=1,x=.01,y=.02,angle=-179)], balls=[dict(x=.32,y=.03)])
            path.write_text('\n'.join(json.dumps(x) for x in [dict(type='sim_scenario',schema=1),a,b]))
            scenario = load_scenario(path)
            self.assertAlmostEqual(scenario.motion[0].omega, 20)
            self.assertAlmostEqual(scenario.motion[1].vy, .3)
            b['time'] = 1
            path.write_text('\n'.join(json.dumps(x) for x in [dict(type='sim_scenario',schema=1),a,b]))
            with self.assertRaises(ValueError): load_scenario(path)

    def test_simulated_command_and_camera_delays_and_stop_cancellation(self):
        with VSSSSim(config=dict(latency=dict(udp_ms=100,ble_ms=80))) as sim:
            sim.cars.set_velocity(1, Velocity(.5))
            for _ in range(5): sim.step(.01)
            self.assertEqual(sim._world.cars[0].pose.x, 0)
            for _ in range(10): sim.step(.01)
            self.assertGreater(sim._world.cars[0].pose.x, 0)
            self.assertLess(sim.gps.snapshot().sequence, sim._world.sequence)
            self.assertGreaterEqual(sim.gps.snapshot().source_age, .1)
            sim.cars.go_to(1, Pose(.2,0))
            sim.cars.stop(1)
            self.assertEqual(sim._pending, {})
            for _ in range(10): sim.step(.01)
            self.assertNotIn(1, sim._navigation)

    def test_seeded_wobble_is_deterministic_and_physics_changes_are_effective(self):
        state = WorldState(balls=[BallState(0)])
        scenario = SimScenario(state, [BodyMotion('ball',0,.3,.1)],
                               dict(ball_wobble=dict(enabled=True)), seed=42)
        with VSSSSim(scenario) as a, VSSSSim(scenario) as b:
            for _ in range(80): a.step(.01); b.step(.01)
            self.assertEqual(a._world.balls, b._world.balls)
        for deceleration, expected in [(.1, .35), (.4, .05)]:
            with VSSSSim(SimScenario(state, [BodyMotion('ball',0,.4,0)],
                                     dict(ball_model=dict(rolling_deceleration=deceleration,stop_speed=0)))) as sim:
                for _ in range(100): sim.step(.01)
                self.assertAlmostEqual(sim._physics.balls[0].velocity.x, max(0,.4-deceleration), delta=.001)

    def test_invalid_configs_fail_without_mutating_running_world(self):
        with VSSSSim() as sim:
            before = sim._world
            for config in [dict(physics_hz=0),dict(body=[[0,0],[1,0],[0,1]]),dict(latency=dict(ble_ms=-1))]:
                with self.assertRaises(ValueError): sim.load_scenario(SimScenario(before, config=config))
                self.assertEqual(sim._world, before)
