"""Small rigid-body world in meters, kilograms and seconds, backed by Pymunk."""
from dataclasses import replace
import math
import random

import pymunk

from .models import BallState, CarState, Pose, Velocity
from .scenarios import physical_config

CAR_SIZE = .12
BALL_RADIUS = .02
PHYSICS_DT = 1 / 240


class Physics:
    def __init__(self, state, config=None, seed=None):
        self.config = physical_config(config)
        self.random = random.Random(seed)
        self.held, self._wobble = set(), {}
        self._impacts = {}
        self.dt = 1 / self.config['physics_hz']
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.iterations = 20
        self.space.collision_slop = .0005
        self.space.on_collision(1, 2, pre_solve=self._ball_contact, post_solve=self._ball_impact)
        self.space.on_collision(2, 3, pre_solve=self._wall_contact)
        self.cars, self.balls = {}, {}
        self._remaining = 0.
        w, h = state.field.width / 2, state.field.height / 2
        corners = [(-w, -h), (w, -h), (w, h), (-w, h)]
        for a, b in zip(corners, corners[1:] + corners[:1]):
            shape = pymunk.Segment(self.space.static_body, a, b, 0)
            shape.friction = .4
            shape.elasticity = self.config['wall_restitution']
            shape.collision_type = 3
            self.space.add(shape)
        for objects, target in ((state.cars, self.cars), (state.balls, self.balls)):
            for obj in objects:
                self._add(obj, target)

    def _add(self, obj, target):
        car = target is self.cars
        mass = self.config['car_mass' if car else 'ball_mass']
        moment = (pymunk.moment_for_poly(mass, self.config['body']) if car
                  else pymunk.moment_for_circle(mass, 0, self.config['ball_radius']))
        body = pymunk.Body(mass, moment)
        body.position = obj.pose.x, obj.pose.y
        body.angle = math.radians(obj.pose.angle)
        shape = (pymunk.Poly(body, self.config['body']) if car
                 else pymunk.Circle(body, self.config['ball_radius']))
        shape.friction, shape.elasticity = (.6, .15) if car else (.3, 0.)
        shape.collision_type = 1 if car else 2
        self.space.add(body, shape)
        target[obj.id] = body

    def _wall_contact(self, arbiter, space, data):
        arbiter.restitution = self.config['wall_restitution']

    def _ball_contact(self, arbiter, space, data):
        car, ball = arbiter.shapes
        normal = arbiter.normal
        point = arbiter.contact_point_set.points[0].point_a
        surface = car.body.velocity_at_world_point(point)
        velocity = ball.body.velocity
        impulse = self.config['ball_model']['impact_gain'] * max(0., (surface-velocity).dot(normal))
        self._impacts[(car, ball)] = (normal, velocity.dot(normal)+impulse)
        arbiter.restitution = 0.

    def _ball_impact(self, arbiter, space, data):
        car, ball = arbiter.shapes
        normal, speed = self._impacts.pop((car, ball))
        # Preserve the identified model's normal impulse, while Pymunk resolves
        # geometry, car reaction and tangential contact/friction.
        ball.body.velocity += normal * (speed-ball.body.velocity.dot(normal))

    def edit(self, kind, obj):
        target = self.cars if kind == 'car' else self.balls
        body = target[obj.id]
        # Recreate the edited body to discard cached contact/bias impulses too.
        self.space.remove(*body.shapes, body)
        self._add(obj, target)
        if (kind, obj.id) in self.held: self.hold(kind, obj.id, True)

    def hold(self, kind, number, held):
        body = (self.cars if kind == 'car' else self.balls)[number]
        key = (kind, number)
        body.velocity, body.angular_velocity = (0, 0), 0
        body.force, body.torque = (0, 0), 0
        if held:
            self.held.add(key)
            body.body_type = pymunk.Body.KINEMATIC
        else:
            self.held.discard(key)
            body.body_type = pymunk.Body.DYNAMIC
            body.mass = self.config['car_mass' if kind == 'car' else 'ball_mass']
            body.moment = (pymunk.moment_for_poly(body.mass, self.config['body']) if kind == 'car'
                           else pymunk.moment_for_circle(body.mass, 0, self.config['ball_radius']))
        self.space.reindex_shapes_for_body(body)

    def initial_motion(self, motions):
        for motion in motions:
            body = (self.cars if motion.kind == 'car' else self.balls)[motion.id]
            body.velocity = motion.vx, motion.vy
            body.angular_velocity = math.radians(motion.omega)

    def advance(self, dt, commands):
        self._remaining += dt
        motion, model, wobble = self.config['motion'], self.config['ball_model'], self.config['ball_wobble']
        while self._remaining + 1e-12 >= self.dt:
            for car_id, body in self.cars.items():
                if ('car', car_id) in self.held: continue
                command = commands.get(car_id, Velocity())
                target = pymunk.Vec2d(command.x, command.y).rotated(body.angle) * motion['max_speed']
                delta = target - body.velocity
                # Finite motor force lets contacts block the car and push balls.
                acceleration = delta / motion['response_time']
                if acceleration.length > motion['max_acceleration']:
                    acceleration = acceleration.normalized() * motion['max_acceleration']
                body.force = acceleration * body.mass
                angular_acceleration = max(-4 * math.pi, min(4 * math.pi,
                    (command.w * math.radians(motion['max_yaw_rate']) - body.angular_velocity) / motion['response_time']))
                body.torque = angular_acceleration * body.moment
            for ball_id, body in self.balls.items():
                if ('ball', ball_id) in self.held: continue
                speed = body.velocity.length
                stop = wobble['stop_speed'] if wobble['enabled'] else model['stop_speed']
                if speed < stop:
                    body.velocity = (0, 0)
                elif speed:
                    drift = pymunk.Vec2d(model['drift_acceleration_x'], model['drift_acceleration_y'])
                    variation = 1.
                    if wobble['enabled']:
                        previous = self._wobble.get(ball_id, pymunk.Vec2d(0, 0))
                        alpha = 1 - math.exp(-self.dt / wobble['correlation_time'])
                        noise = pymunk.Vec2d(self.random.uniform(-1, 1), self.random.uniform(-1, 1))
                        previous = previous * (1-alpha) + noise * alpha
                        self._wobble[ball_id] = previous
                        if speed >= wobble['onset_speed']: drift += previous * wobble['acceleration']
                        variation += previous.x * wobble['drag_variation']
                        if speed < wobble['onset_speed']: variation *= wobble['slow_drag_scale']
                    body.velocity *= max(0., speed - model['rolling_deceleration'] * variation * self.dt) / speed
                    body.velocity += drift * self.dt
                body.angular_velocity *= math.exp(-3 * self.dt)
            self.space.step(self.dt)
            self._remaining -= self.dt

    def snapshot(self, state):
        def pose(body):
            return Pose(body.position.x, body.position.y, (math.degrees(body.angle) + 180) % 360 - 180)
        return replace(state, cars=[CarState(car.id, pose(self.cars[car.id])) for car in state.cars],
                       balls=[BallState(ball.id, pose(self.balls[ball.id])) for ball in state.balls])
