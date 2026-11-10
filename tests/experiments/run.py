"""Run with python -m tests.experiments.run. Every phase is recorded, including aborts."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

from car import Car
from gps import GPS
from mecanum_client import normalize_trim_matrix
from .recording import Recorder
from .plans import Pulse, Segment, Stride, build_plan


from .session import Experiment
from .scenarios import SCENARIOS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Enable physical commands; default only previews')
    parser.add_argument('--observe', type=float, help='Record telemetry for this many seconds without BLE')
    parser.add_argument('--experiment', choices=('strides','correction','waypoints'), default='strides')
    parser.add_argument('--controller', choices=('p','pid','predictive'), default='predictive')
    parser.add_argument('--telemetry', choices=('te2004b','vsss'), default='te2004b')
    parser.add_argument('--telemetry-port', type=int, help='Default TE2004B 5000, VSSS 45888')
    parser.add_argument('--stale-after', type=float, default=.5)
    parser.add_argument('--telemetry-recovery', type=float, default=5., help='Safe stopped wait for telemetry outside active pulses')
    parser.add_argument('--trim-file', type=Path, help='Candidate 3x3 matrix; omit with --auto-fit')
    parser.add_argument('--auto-fit', action='store_true', help='Fit the correction matrix from the identity half of a correction run')
    parser.add_argument('--keep-trim', action='store_true', help='Leave the validated candidate active after a correction run')
    parser.add_argument('--waypoints', type=Path, help='Waypoint JSON; default is a small rectangle')
    parser.add_argument('--car', type=int, default=1)
    parser.add_argument('--connect-timeout', type=float, default=30.)
    parser.add_argument('--profile', choices=('quick', 'standard', 'dynamic'), default='quick')
    parser.add_argument('--stride-scale', type=float, default=1., help='Scale dynamic stride segment durations')
    parser.add_argument('--recenter-every', type=int, help='Default: 2 strides for dynamic, 1 otherwise')
    parser.add_argument('--bounds-x', type=float, help='Absolute X bound in metres; dynamic default .8')
    parser.add_argument('--bounds-y', type=float, help='Absolute Y bound in metres; dynamic default 1.5')
    parser.add_argument('--boundary-margin', type=float, default=.2)
    parser.add_argument('--lookahead', type=float, default=.6, help='Predictive boundary horizon, seconds')
    parser.add_argument('--levels', type=float, nargs='+')
    parser.add_argument('--axes', choices=('x', 'y', 'w'), nargs='+', default=['x', 'y', 'w'])
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--start-trial', type=int, default=1, help='Resume a plan in a NEW recording directory')
    parser.add_argument('--duration', type=float, default=.75)
    parser.add_argument('--settle', type=float, default=1.)
    parser.add_argument('--period', type=float, default=.05)
    parser.add_argument('--center-x', type=float, default=0.)
    parser.add_argument('--center-y', type=float, default=.75)
    parser.add_argument('--heading', type=float, help='Degrees; default preserves the initial heading')
    parser.add_argument('--center-timeout', type=float, default=45.)
    parser.add_argument('--center-hold', type=float, default=.5)
    parser.add_argument('--center-speed', type=float, default=.4)
    parser.add_argument('--center-w', type=float)
    parser.add_argument('--min-center-speed', type=float)
    parser.add_argument('--min-center-w', type=float)
    parser.add_argument('--position-tolerance', type=float, default=.025)
    parser.add_argument('--heading-tolerance', type=float, default=5.)
    parser.add_argument('--max-radius', type=float, default=.35, help='Metres from the chosen center')
    parser.add_argument('--max-seconds', type=float, default=600.)
    parser.add_argument('--kp-dist', type=float)
    parser.add_argument('--ki-dist', type=float, default=0.)
    parser.add_argument('--kd-dist', type=float, default=.2)
    parser.add_argument('--kp-heading', type=float)
    parser.add_argument('--ki-heading', type=float, default=0.)
    parser.add_argument('--kd-heading', type=float, default=.1)
    parser.add_argument('--prediction-horizon', type=float, default=.2)
    parser.add_argument('--velocity-alpha', type=float, default=.5)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--list-plan', action='store_true', help='Print the test plan without connecting or moving')
    args = parser.parse_args(argv)
    predictive = args.controller == 'predictive'
    args.center_w = (.35 if predictive else .5) if args.center_w is None else args.center_w
    args.min_center_speed = (.12 if predictive else 0.) if args.min_center_speed is None else args.min_center_speed
    args.min_center_w = (.08 if predictive else 0.) if args.min_center_w is None else args.min_center_w
    args.kp_dist = (4. if predictive else 5.) if args.kp_dist is None else args.kp_dist
    args.kp_heading = (.55 if predictive else .8) if args.kp_heading is None else args.kp_heading
    if args.execute and args.observe is not None:
        parser.error('observe does not allow execute')
    if args.telemetry_port is None:
        args.telemetry_port = 5000 if args.telemetry == 'te2004b' else 45888
    if not 0 <= args.telemetry_port <= 65534:
        parser.error('telemetry-port must be between 0 and 65534')
    args.trim_matrix = None
    if args.experiment == 'correction':
        if args.auto_fit and args.trim_file is not None:
            parser.error('Use either --auto-fit or --trim-file, not both')
        if not args.auto_fit and args.trim_file is None:
            parser.error('correction comparison requires --trim-file or --auto-fit')
        if args.trim_file is not None:
            try:
                specification = json.loads(args.trim_file.read_text())
                args.trim_matrix = normalize_trim_matrix(specification['matrix'] if isinstance(specification,dict) else specification)
            except (OSError, ValueError, TypeError, KeyError) as error:
                parser.error(f'Invalid trim file: {error}')
    elif args.trim_file is not None:
        parser.error('trim-file is only used by the correction comparison')
    elif args.auto_fit or args.keep_trim:
        parser.error('auto-fit and keep-trim are only used by correction experiments')
    args.waypoint_targets = []
    if args.experiment == 'waypoints':
        args.waypoints = args.waypoints or Path(__file__).with_name('configs')/'waypoints.json'
        try:
            targets = json.loads(args.waypoints.read_text())
            for target in targets:
                if not isinstance(target,list) or len(target) not in (2,3):
                    raise ValueError('waypoints need x, y, and optional heading in degrees')
                x,y = float(target[0]),float(target[1])
                heading = None if len(target)==2 or target[2] is None else float(target[2])
                if not all(math.isfinite(v) for v in (x,y)) or (heading is not None and not math.isfinite(heading)):
                    raise ValueError('waypoints must be finite')
                args.waypoint_targets.append((x,y,heading))
            if not args.waypoint_targets:
                raise ValueError('waypoint plan is empty')
        except (OSError, ValueError, TypeError) as error:
            parser.error(f'Invalid waypoints: {error}')
    if args.recenter_every is None:
        args.recenter_every = 2 if args.profile == 'dynamic' else 1
    if args.profile == 'dynamic':
        if args.levels is not None:
            parser.error('The dynamic profile defines its own multi-axis commands')
    if args.profile == 'dynamic' or args.experiment == 'waypoints':
        args.bounds_x = .8 if args.bounds_x is None else args.bounds_x
        args.bounds_y = 1.5 if args.bounds_y is None else args.bounds_y
    if (args.bounds_x is None) != (args.bounds_y is None):
        parser.error('Specify both bounds-x and bounds-y')
    for name, value in vars(args).items():
        if isinstance(value, float) and not math.isfinite(value):
            parser.error(f'{name} must be finite')
    positive = ('duration', 'settle', 'period', 'connect_timeout', 'center_timeout', 'center_hold',
                'center_speed', 'center_w', 'position_tolerance', 'heading_tolerance',
                'max_radius', 'max_seconds', 'repeats', 'recenter_every', 'stride_scale', 'lookahead', 'boundary_margin', 'stale_after',
                'telemetry_recovery')
    if any(getattr(args, name) <= 0 for name in positive):
        parser.error('Durations, limits, tolerances, and repeats must be positive')
    if args.observe is not None and args.observe <= 0:
        parser.error('observe duration must be positive')
    if args.bounds_x is not None and (args.bounds_x <= abs(args.center_x)+args.boundary_margin
                                    or args.bounds_y <= abs(args.center_y)+args.boundary_margin):
        parser.error('Center and boundary margin must fit inside the rectangular bounds')
    if any(abs(x)>args.bounds_x-args.boundary_margin or abs(y)>args.bounds_y-args.boundary_margin
           for x,y,_ in args.waypoint_targets):
        parser.error('Waypoints must fit inside the inner rectangular boundary')
    if args.car < 0 or not 0 <= args.min_center_speed <= args.center_speed <= 1 or not 0 <= args.min_center_w <= args.center_w <= 1:
        parser.error('Invalid car number or normalized centering limits')
    if args.prediction_horizon < 0 or not 0 < args.velocity_alpha <= 1:
        parser.error('Prediction horizon must be nonnegative and velocity-alpha in (0, 1]')
    if args.levels and any(not math.isfinite(level) or not 0 < level <= 1 for level in args.levels):
        parser.error('Levels must be within (0, 1]')
    if not 1 <= args.start_trial <= len(build_plan(args)):
        parser.error('start-trial must select an existing trial in the plan')
    return args


def main(argv=None):
    args = parse_args(argv)
    plan = build_plan(args)
    configuration = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    if args.list_plan:
        print(json.dumps([asdict(item) for item in plan], indent=2))
        return 0
    if not args.execute and args.observe is None:
        print(json.dumps(dict(mode='preview',config=configuration,plan=[asdict(item) for item in plan]),indent=2))
        print('Preview only: no BLE connection, trim changes, or motor commands. Use --observe SECONDS for passive recording; --execute enables movement.')
        return 0
    directory = args.output or Path('tests/experiments/runs') / (time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+f'_car{args.car}')
    directory.mkdir(parents=True,exist_ok=False)
    print(f'RECORDING: {directory.resolve()}',flush=True)
    print(f'To stop: Ctrl+C or create {directory.resolve() / "STOP"}',flush=True)
    metadata = dict(config=configuration,plan=[asdict(item) for item in plan],
                    mode='execute' if args.execute else 'observe',wall_time_ns=time.time_ns(),
                    clock_resolution_s=time.get_clock_info('monotonic').resolution,
                    python=sys.version)
    (directory/'metadata.json').write_text(json.dumps(metadata,indent=2))
    gps=None
    try:
        gps=GPS(port=args.telemetry_port,stale_after=args.stale_after,subscribe=args.telemetry=='vsss')
        if args.observe is not None:
            from .observation import observe
            return observe(args,directory,gps)
        with Car(args.car,timeout=args.connect_timeout) as car:
            target=[args.center_x,args.center_y,args.heading]
            recorder=Recorder(directory,car,gps,target)
            original_matrix=car.trim_matrix
            try:
                pose=car.wait_for_position()
                if target[2] is None:
                    target[2]=pose.angle
                metadata.update(target=target,trim_matrix=original_matrix,
                                resolution=car._client.resolution,monotonic_start=recorder.started)
                experiment=SCENARIOS[args.experiment](car,gps,recorder,args,directory)
                controller=getattr(experiment,'controller',None)
                metadata['effective_controller']=dict(name=args.controller,type=type(controller).__name__)
                for axis in ('distance','heading'):
                    pid=getattr(controller,axis,None)
                    if pid is not None:
                        metadata['effective_controller'][axis]=dict(gains=pid.tunings,output_limits=pid.output_limits)
                (directory/'metadata.json').write_text(json.dumps(metadata,indent=2))
                experiment.run(plan)
            finally:
                try:
                    if args.experiment=='correction':
                        if (args.keep_trim and getattr(experiment, 'completed', False)
                                and getattr(experiment, 'candidate_matrix', None) is not None):
                            recorder.record('trim_kept', note='validated candidate remains active')
                        elif original_matrix is not None and car.is_connected:
                            car.stop()
                            car.set_trim(original_matrix)
                            recorder.record('trim_restored')
                        else:
                            recorder.record('trim_restore_unavailable',note='Connection or original matrix unavailable')
                finally:
                    recorder.close()
    except BaseException as error:
        if not (directory/'result.json').exists():
            result=dict(status='aborted',phase='startup',reason=f'{type(error).__name__}: {error}')
            (directory/'result.json').write_text(json.dumps(result,indent=2))
            (directory/'status.json').write_text(json.dumps(result,indent=2))
        raise
    finally:
        if gps is not None:
            gps.close()
    print(f'Completed: {directory.resolve() / "samples.csv"}',flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Interrupted; car stopped and partial CSV retained.', flush=True)
        raise SystemExit(130)
    except Exception as error:
        print(f'ABORTED: {error}', flush=True)
        raise SystemExit(1)
