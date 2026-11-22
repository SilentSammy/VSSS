import math
import unittest

from vsss import BallState, CarState, Field, Pose, StatePatch, Velocity, WorldState
from vsss.physics import Physics
from vsss.scenarios import DEFAULT_CONFIG
from vsss.simulation import VSSSSim


class PhysicsTest(unittest.TestCase):
    def test_calibrated_legacy_defaults_are_preserved(self):
        self.assertEqual(DEFAULT_CONFIG['body'], [[-.07, -.07], [.07, -.07], [.07, .07], [-.07, .07]])
        self.assertAlmostEqual(DEFAULT_CONFIG['ball_model']['impact_gain'], .8803656717213731)
        self.assertAlmostEqual(DEFAULT_CONFIG['ball_model']['rolling_deceleration'], .07489989178563962)
        self.assertTrue(DEFAULT_CONFIG['ball_wobble']['enabled'])
        self.assertEqual(DEFAULT_CONFIG['physics_hz'], 200)

    def test_legacy_free_roll_distance_and_low_speed_wobble_region(self):
        state = WorldState(field=Field(8, 8), balls=[BallState(0)])
        config = dict(ball_model=dict(impact_gain=1., rolling_deceleration=.1,
                                     drift_acceleration_x=0., drift_acceleration_y=0., stop_speed=0.),
                      ball_wobble=dict(enabled=False))
        physics = Physics(state, config, seed=1)
        physics.balls[0].velocity = .4, 0
        for _ in range(500): physics.advance(.01, {})
        self.assertAlmostEqual(physics.balls[0].position.x, .4 ** 2 / (2 * .1), delta=.003)

        def slow(seed):
            value = Physics(state, dict(ball_model=dict(drift_acceleration_x=0., drift_acceleration_y=0.)), seed=seed)
            value.balls[0].velocity = .1, 0
            for _ in range(100): value.advance(.01, {})
            return tuple(value.balls[0].position)
        self.assertEqual(slow(1), slow(1))
        self.assertNotEqual(slow(1), slow(2))

    def test_push_ball_then_coast_and_roll_to_stop(self):
        with VSSSSim(WorldState(cars=[CarState(1, Pose(-.25, 0))], balls=[BallState(0)])) as sim:
            sim.cars.set_velocity(1, Velocity(.5))
            for _ in range(100): sim.step(.01)
            self.assertGreater(sim.gps.state().balls[0].pose.x, .06)
            self.assertGreater(sim._physics.balls[0].velocity.x, .1)
            # Reset only the car; a freely rolling ball retains momentum.
            sim.update_state(StatePatch(cars=[CarState(1, Pose(-.4, .4))]))
            for _ in range(1000): sim.step(.01)
            self.assertLess(sim._physics.balls[0].velocity.length, .011)

    def test_wall_bounce_and_car_contacts(self):
        with VSSSSim(WorldState(balls=[BallState(0, Pose(.65, 0))])) as sim:
            body = sim._physics.balls[0]
            body.velocity = 2, 0
            for _ in range(20): sim.step(.01)
            self.assertLess(body.velocity.x, 0)
            self.assertLess(abs(body.velocity.x), 2)
            self.assertLess(sim.gps.state().balls[0].pose.x, .73)
        with VSSSSim(WorldState(cars=[CarState(1, Pose(-.25, 0)), CarState(2)])) as sim:
            sim.cars.set_velocity(1, Velocity(.5))
            for _ in range(400): sim.step(.01)
            cars = sim.gps.state().cars
            self.assertGreater(cars[1].pose.x, .01)
            self.assertGreater(cars[1].pose.x - cars[0].pose.x, .115)
            self.assertLess(cars[1].pose.x, .695)

    def test_body_frame_and_fixed_substeps(self):
        initial = WorldState(cars=[CarState(1, Pose(0, 0, 90))])
        with VSSSSim(initial) as a, VSSSSim(initial) as b:
            a.cars.set_velocity(1, Velocity(.2))
            b.cars.set_velocity(1, Velocity(.2))
            for _ in range(30): a.step(1/30)
            for _ in range(100): b.step(.01)
            pose = a.gps.state().cars[0].pose
            self.assertAlmostEqual(pose.x, 0, places=6)
            self.assertGreater(pose.y, .1)
            self.assertAlmostEqual(pose.y, b.gps.state().cars[0].pose.y, places=8)

    def test_edits_and_reload_clear_physical_momentum(self):
        initial = WorldState(cars=[CarState(1)], balls=[BallState(0, Pose(.3, .3))])
        with VSSSSim(initial) as sim:
            sim.cars.set_velocity(1, Velocity(.5))
            sim.step(.1)
            sim._physics.balls[0].velocity = 1, 1
            sim.update_state(StatePatch(balls=[BallState(0, Pose(.2, .2))]))
            self.assertEqual(sim._physics.balls[0].velocity.length, 0)
            self.assertTrue(sim._velocities)
            sim.load_scenario(initial)
            sim.step(.1)
            self.assertEqual(sim.gps.state().cars, initial.cars)
            self.assertEqual(sim.gps.state().balls, initial.balls)
