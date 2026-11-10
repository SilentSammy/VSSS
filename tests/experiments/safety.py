"""Common geometric limits for strides, correction comparisons and waypoint runs."""
import math


class SafetyPolicy:
    def __init__(self, args):
        self.args = args

    def check(self, pose):
        a = self.args
        if a.bounds_x is not None:
            if abs(pose.x) > a.bounds_x or abs(pose.y) > a.bounds_y:
                raise RuntimeError('Hard rectangular experiment boundary exceeded')
        elif math.hypot(pose.x-a.center_x, pose.y-a.center_y) > a.max_radius:
            raise RuntimeError('Experiment boundary exceeded')

    def approaching(self, pose, velocity):
        a = self.args
        if a.bounds_x is None:
            return False
        predicted = (pose.x+velocity[0]*a.lookahead, pose.y+velocity[1]*a.lookahead)
        return any(abs(position) >= bound-a.boundary_margin
                   for point in ((pose.x,pose.y),predicted)
                   for position,bound in zip(point,(a.bounds_x,a.bounds_y)))
