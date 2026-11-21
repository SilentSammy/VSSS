"""Deterministic model checks, with no sockets or hardware."""
import copy
import math
import unittest

from vsss._sim import (BallModel, SimWorld, ball_contact, load_config,
                       polygon, polygon_contact)


class SimPhysicsTests(unittest.TestCase):
    def world(self, **overrides):
        config = load_config()
        config.update(overrides)
        return SimWorld(config)

    def advance(self, world, seconds):
        for _ in range(round(seconds * world.physics_hz)):
            world.step(1 / world.physics_hz)

    def test_body_frame_motion_and_mecanum_saturation(self):
        world = self.world(balls=[])
        car = world.cars[1]
        car.command = (1., 0., 0.)
        self.advance(world, 1.)
        self.assertAlmostEqual(car.x, 0., places=6)
        self.assertAlmostEqual(car.y, -.45 + .6 * (1 - .12), delta=.005)
        car.stop()
        car.angle = 0.
        car.command = (1., 1., 0.)
        self.advance(world, .5)
        self.assertAlmostEqual(car.vx, .3, delta=.01)
        self.assertAlmostEqual(car.vy, .3, delta=.01)

    def test_trim_and_alternative_body_shape(self):
        world = self.world(balls=[], body=[[-.08, -.04], [.08, -.04], [0, .08]])
        car = world.cars[1]
        car.angle = 0
        car.trim = (.5, 0, 0, 0, 1, 0, 0, 0, 1)
        car.command = (1., 0., 0.)
        self.advance(world, .8)
        self.assertAlmostEqual(car.vx, .3, delta=.002)
        self.assertEqual(len(world.vertices(car)), 3)

    def test_contact_substeps_scale_with_requested_step(self):
        config = dict(load_config(), balls=[], physics_hz=50, ball_radius=.002,
                      motion=dict(max_speed=2, max_yaw_rate=180, response_time=.12))
        a, b = SimWorld(config), SimWorld(config)
        for world in (a, b):
            world.cars[1].command = (.5, 0, .1)
        a.step(.1)
        for _ in range(5):
            b.step(.02)
        self.assertAlmostEqual(a.cars[1].x, b.cars[1].x, places=10)
        self.assertAlmostEqual(a.cars[1].y, b.cars[1].y, places=10)

    def test_repeatability_and_stationary_ball_has_no_drift(self):
        a, b = self.world(random_seed=123), self.world(random_seed=123)
        for world in (a, b):
            world.cars[1].command = (.5, .2, .1)
            self.advance(world, .6)
        self.assertEqual(a.packet(), b.packet())
        self.assertEqual((a.balls[0].x, a.balls[0].y), (0., 0.))

    def test_free_roll_matches_analytic_stop(self):
        world = self.world(cars=[dict(id=1, x=0, y=-1, angle=0)],
                           ball_wobble=dict(enabled=False),
                           ball_model=dict(impact_gain=1., rolling_deceleration=.5, stop_speed=.05),
                           balls=[dict(x=-.4, y=0, vx=.6)])
        self.advance(world, 3)
        ball = world.balls[0]
        self.assertAlmostEqual(ball.x, -.4 + (.6 ** 2 - .05 ** 2) / 1., delta=.003)
        self.assertEqual((ball.vx, ball.vy), (0., 0.))
        x, y = ball.x, ball.y
        self.advance(world, 1)
        self.assertEqual((ball.x, ball.y), (x, y))

    def rolling_world(self, seed=1, speed=.1, **settings):
        config = load_config()
        config['random_seed'] = seed
        config['cars'] = [dict(id=1, x=0, y=-1, angle=0)]
        config['balls'] = [dict(x=-.2, y=0, vx=speed)]
        config['ball_wobble'].update(settings)
        return SimWorld(config)

    def test_wobble_repeats_by_seed_and_different_seeds_change_trajectory(self):
        a, b, c = self.rolling_world(1), self.rolling_world(1), self.rolling_world(2)
        for world in (a, b, c):
            self.advance(world, 1)
        self.assertEqual(a.packet(), b.packet())
        self.assertNotEqual((a.balls[0].x, a.balls[0].y), (c.balls[0].x, c.balls[0].y))

    def test_wobble_never_adds_speed_and_ball_remains_at_rest(self):
        for seed in range(8):
            world = self.rolling_world(seed)
            ball = world.balls[0]
            speed = math.hypot(ball.vx, ball.vy)
            for _ in range(1200):
                world.step(.005)
                next_speed = math.hypot(ball.vx, ball.vy)
                self.assertLessEqual(next_speed, speed + 1e-12)
                speed = next_speed
            self.assertEqual(speed, 0)
            position = ball.x, ball.y
            self.advance(world, .5)
            self.assertEqual((ball.x, ball.y), position)

    def test_wobble_only_changes_slow_motion_and_disable_restores_base_model(self):
        a, b = self.rolling_world(1, speed=.4), self.rolling_world(2, speed=.4)
        self.advance(a, .2)
        self.advance(b, .2)
        self.assertEqual(a.packet(), b.packet())
        a, b = self.rolling_world(1, enabled=False), self.rolling_world(2, enabled=False)
        self.advance(a, 2)
        self.advance(b, 2)
        self.assertEqual(a.packet(), b.packet())
        expected = a.model.stopping_point((-.2, 0), (.1, 0))
        self.assertAlmostEqual(a.balls[0].x, expected[0], delta=.002)
        self.assertAlmostEqual(a.balls[0].y, expected[1], delta=.002)

    def test_slow_roll_extends_past_old_five_centimetre_per_second_cutoff(self):
        world = self.rolling_world(1, speed=.04)
        ball = world.balls[0]
        self.advance(world, 2)
        self.assertGreater(math.hypot(ball.x + .2, ball.y), .01)
        self.assertEqual((ball.vx, ball.vy), (0, 0))

    def test_random_seed_is_recorded_and_does_not_mutate_caller_configuration(self):
        config = load_config()
        self.assertIsNone(config['random_seed'])
        world = SimWorld(config)
        self.assertIsNone(config['random_seed'])
        self.assertIsInstance(world.seed, int)
        self.assertEqual(world.setup()['simulation']['config']['random_seed'], world.seed)

    def test_invalid_wobble_configuration_is_rejected(self):
        for settings in [dict(enabled=1), dict(correlation_time=0), dict(acceleration=-1),
                         dict(stop_speed=.2), dict(drag_variation=2), dict(slow_drag_scale=0)]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.rolling_world(**settings)
        with self.assertRaises(ValueError):
            self.rolling_world(seed=-1)

    def test_held_and_repositioned_balls_clear_wobble_without_advancing_random_stream(self):
        world = self.rolling_world()
        self.advance(world, .2)
        ball = world.balls[0]
        world.edit('editor', dict(operation='grab', kind='ball', id=0))
        world.edit('editor', dict(operation='move', kind='ball', id=0, x=.3, y=.4))
        state = ball.rng.getstate()
        self.advance(world, .5)
        self.assertEqual((ball.x, ball.y, ball.wobble_lateral, ball.wobble_drag), (.3, .4, 0, 0))
        self.assertEqual(ball.rng.getstate(), state)
        world.edit('editor', dict(operation='release', kind='ball', id=0))
        self.advance(world, .5)
        self.assertEqual((ball.x, ball.y), (.3, .4))

    def test_other_balls_have_independent_random_streams(self):
        def make(other_speed):
            config = self.rolling_world().config
            config['balls'].append(dict(x=.4, y=.6, vx=other_speed))
            return SimWorld(config)
        a, b = make(0), make(.08)
        self.advance(a, 1)
        self.advance(b, 1)
        self.assertEqual((a.balls[0].x, a.balls[0].y), (b.balls[0].x, b.balls[0].y))

    def test_convex_geometry_faces_corners_and_inside(self):
        vertices = polygon(load_config()['body'])
        face = ball_contact(vertices, (.08, 0), .02)
        self.assertAlmostEqual(face[1], .01)
        self.assertEqual(face[0], (1., 0.))
        corner = ball_contact(vertices, (.08, .08), .02)
        self.assertAlmostEqual(corner[0][0], math.sqrt(.5))
        self.assertAlmostEqual(corner[0][1], math.sqrt(.5))
        self.assertGreater(ball_contact(vertices, (0, 0), .02)[1], .02)
        self.assertIsNone(ball_contact(vertices, (.10, .10), .02))
        self.assertIsNotNone(polygon_contact(vertices, vertices))

    def test_car_hits_ball_then_ball_rolls_to_rest(self):
        world = self.world()
        car, ball = world.cars[1], world.balls[0]
        car.command = (.7, 0., 0.)
        self.advance(world, 1.)
        self.assertGreater(ball.vy, .1)
        self.assertGreater(ball.y, 0.)
        car.stop()
        self.advance(world, 10.)
        self.assertEqual((ball.vx, ball.vy), (0., 0.))
        self.assertLessEqual(abs(ball.y), world.height / 2 - world.radius + 1e-9)

    def test_rotating_surface_transfers_tangential_velocity(self):
        world = self.world(cars=[dict(id=1, x=0, y=0, angle=0)],
                           balls=[dict(x=.095, y=.06)])
        car, ball = world.cars[1], world.balls[0]
        car.command = (0., 0., -1.)
        self.advance(world, .15)
        self.assertGreater(math.hypot(ball.vx, ball.vy), .02)

    def test_car_walls_and_car_contacts_prevent_pass_through(self):
        world = self.world(balls=[], cars=[dict(id=1, x=-.2, y=0, angle=0),
                                          dict(id=2, x=.2, y=0, angle=180)])
        for car in world.cars.values():
            car.command = (1., 0., 0.)
        self.advance(world, 2.)
        a, b = world.cars.values()
        self.assertLess(a.x, b.x)
        self.assertGreaterEqual(b.x - a.x, .14 - 1e-7)
        b.stop()
        a.command = (-1., 0., 0.)
        self.advance(world, 3.)
        self.assertGreaterEqual(min(x for x, y in world.vertices(a)), -.8 - 1e-9)

    def test_fast_ball_bounces_off_wall_and_car(self):
        world = self.world(cars=[dict(id=1, x=0, y=0, angle=45)],
                           balls=[dict(x=-.5, y=0, vx=5)])
        self.advance(world, .2)
        ball = world.balls[0]
        self.assertIsNone(ball_contact(world.vertices(world.cars[1]), (ball.x, ball.y), world.radius - 1e-8))
        self.assertLess(ball.x, .1)
        world = self.world(balls=[dict(x=.7, y=0, vx=3)])
        self.advance(world, .1)
        self.assertLess(world.balls[0].vx, 0)

    def test_ball_ball_collision_and_rest(self):
        world = self.world(balls=[dict(x=-.1, y=0, vx=1), dict(x=.1, y=0)])
        self.advance(world, .3)
        a, b = world.balls
        self.assertGreater(b.vx, a.vx)
        self.assertGreaterEqual(b.x - a.x, .04 - 1e-7)

    def test_firmware_waypoint_arrival_and_timeout(self):
        world = self.world(balls=[])
        car = world.cars[1]
        world.waypoint(car, (.2, .2, 0))
        self.advance(world, 12.)
        self.assertEqual(car.mode, 3)
        self.assertLess(math.hypot(car.x - .2, car.y - .2), .025)
        self.assertLess(abs(car.angle), 5)
        car.controller['waypoint_timeout_ms'] = 1000
        world.waypoint(car, (.8, 1.5, 0))
        self.advance(world, 2.)
        self.assertEqual((car.mode, car.fault), (4, 2))
        self.assertEqual(car.command, (0., 0., 0.))

    def test_rejects_bad_configuration(self):
        baseline = load_config()
        cases = [dict(physics_hz=0), dict(telemetry_hz=10000), dict(field=[0, 1]),
                 dict(body=[[0, 0], [1, 0], [0, 1]]), dict(body=[[0, 0]]),
                 dict(cars=[dict(id=1, x=10, y=0, angle=0)]),
                 dict(cars=[dict(id=1, x=0, y=0, angle=0)] * 2),
                 dict(ball_radius=float('nan')), dict(motion={}),
                 dict(balls=[dict(x=0, y=-.45)]), dict(balls=[dict(x=1, y=0)]),
                 dict(ball_model=dict(impact_gain=1, rolling_deceleration=-1))]
        for overrides in cases:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                SimWorld(dict(copy.deepcopy(baseline), **overrides))


if __name__ == '__main__':
    unittest.main()
