"""Declarative strides and waypoint plans; constructing plans never connects to hardware."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Pulse:
    axis: str
    amplitude: float
    seconds: float

    @property
    def command(self):
        return tuple(self.amplitude if axis == self.axis else 0. for axis in ('x', 'y', 'w'))

    @property
    def name(self):
        return f'{self.axis}={self.amplitude:+.2f}'

    @property
    def segments(self):
        return (Segment(self.command, self.seconds),)


@dataclass(frozen=True)
class Segment:
    command: tuple
    seconds: float


@dataclass(frozen=True)
class Stride:
    name: str
    segments: tuple


@dataclass(frozen=True)
class Waypoint:
    name: str
    target: tuple


def build_plan(args):
    if args.experiment == 'waypoints':
        return [Waypoint(f'waypoint_{i}', tuple(target))
                for i,target in enumerate(args.waypoint_targets, 1)]
    if args.profile == 'dynamic':
        # No stops between a stride's segments: axes enter/leave while moving.
        patterns = [
            ('long_forward', [((.55,0,0),2.2)]),
            ('long_reverse', [((-.55,0,0),2.2)]),
            ('forward_add_left', [((.5,0,0),.9),((.5,.4,0),1.2),((0,.4,0),.9)]),
            ('reverse_add_right', [((-.5,0,0),.9),((-.5,-.4,0),1.2),((0,-.4,0),.9)]),
            ('left_add_yaw', [((0,.55,0),1.),((0,.55,.3),1.3),((0,0,.3),.7)]),
            ('right_add_yaw', [((0,-.55,0),1.),((0,-.55,-.3),1.3),((0,0,-.3),.7)]),
            ('three_axes_yaw_reversal', [((.45,-.3,0),1.),((.45,-.3,.35),1.3),((.45,0,-.35),.7)]),
            ('three_axes_mirrored', [((-.45,.3,0),1.),((-.45,.3,-.35),1.3),((-.45,0,.35),.7)]),
            ('forward_reverse_add_left', [((.6,0,0),1.),((-.6,0,0),1.),((-.6,.3,0),.7)]),
            ('left_right_add_rotation', [((0,.6,0),1.),((0,-.6,0),1.),((.25,-.4,.25),.7)]),
        ]
        plan = [Stride(name, tuple(Segment(command, duration*args.stride_scale) for command,duration in pattern))
                for _ in range(args.repeats) for name,pattern in patterns]
    else:
        levels = args.levels or ([.5] if args.profile == 'quick' else [.3, .5, .7])
        plan = [Pulse(axis, sign * level, args.duration)
                for _ in range(args.repeats) for level in levels
                for axis in args.axes for sign in (1, -1)]
    return plan + plan if args.experiment == 'correction' else plan


