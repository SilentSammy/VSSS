"""Small rigid-body world in meters, kilograms and seconds, backed by Pymunk."""
from dataclasses import replace
import math

import pymunk

from .models import BallState, CarState, Pose, Velocity

CAR_SIZE = .12
BALL_RADIUS = .02
PHYSICS_DT = 1 / 240


class Physics:
    def __init__(self, state):
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.iterations = 20
        self.space.collision_slop = .0005
        self.cars, self.balls = {}, {}
        self._remaining = 0.
        w, h = state.field.width / 2, state.field.height / 2
        corners = [(-w, -h), (w, -h), (w, h), (-w, h)]
        for a, b in zip(corners, corners[1:] + corners[:1]):
            shape = pymunk.Segment(self.space.static_body, a, b, 0)
            shape.friction, shape.elasticity = .4, .8
            self.space.add(shape)
        for objects, target in ((state.cars, self.cars), (state.balls, self.balls)):
            for obj in objects:
                self._add(obj, target)

    def _add(self, obj, target):
        car = target is self.cars
        mass = .4 if car else .045
        moment = (pymunk.moment_for_box(mass, (CAR_SIZE, CAR_SIZE)) if car
                  else pymunk.moment_for_circle(mass, 0, BALL_RADIUS))
        body = pymunk.Body(mass, moment)
        body.position = obj.pose.x, obj.pose.y
        body.angle = math.radians(obj.pose.angle)
        shape = (pymunk.Poly.create_box(body, (CAR_SIZE, CAR_SIZE)) if car
                 else pymunk.Circle(body, BALL_RADIUS))
        shape.friction, shape.elasticity = (.6, .15) if car else (.3, .75)
        self.space.add(body, shape)
        target[obj.id] = body

    def edit(self, kind, obj):
        target = self.cars if kind == 'car' else self.balls
        body = target[obj.id]
        # Recreate the edited body to discard cached contact/bias impulses too.
        self.space.remove(*body.shapes, body)
        self._add(obj, target)

    def advance(self, dt, commands):
        self._remaining += dt
        while self._remaining + 1e-12 >= PHYSICS_DT:
            for car_id, body in self.cars.items():
                command = commands.get(car_id, Velocity())
                target = pymunk.Vec2d(command.x * .6, command.y * .6).rotated(body.angle)
                delta = target - body.velocity
                # Finite motor force lets contacts block the car and push balls.
                acceleration = delta / .06
                if acceleration.length > 3.:
                    acceleration = acceleration.normalized() * 3.
                body.force = acceleration * body.mass
                angular_acceleration = max(-4 * math.pi, min(4 * math.pi,
                    (command.w * math.pi - body.angular_velocity) / .06))
                body.torque = angular_acceleration * body.moment
            for body in self.balls.values():
                speed = body.velocity.length
                if speed:
                    body.velocity *= max(0., speed - .25 * PHYSICS_DT) / speed
                body.angular_velocity *= math.exp(-3 * PHYSICS_DT)
            self.space.step(PHYSICS_DT)
            self._remaining -= PHYSICS_DT

    def snapshot(self, state):
        def pose(body):
            return Pose(body.position.x, body.position.y, (math.degrees(body.angle) + 180) % 360 - 180)
        return replace(state, cars=[CarState(car.id, pose(self.cars[car.id])) for car in state.cars],
                       balls=[BallState(ball.id, pose(self.balls[ball.id])) for ball in state.balls])
