"""Fit/verify linear trim, then fit/verify mixed-axis residual compensation.

Short BLE bursts observed directly through host UDP, with partial checkpoints.
Other markers, field boundaries and onboard pose freshness do not gate motion.
Run stages in order using the same --output directory. --execute permits pulses;
--apply additionally permits quality-gated writes. A failed verification rolls
back its preceding stage. Reposition manually between stages if requested.
"""
import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
from queue import SimpleQueue
import time

from vsss import Car, GPS
from tools.calibrate import Monitor, assess, angle_delta
from tools._fit_trim import fit
from tools._fit_mixed import fit_mixed, PAIRS


class StageMonitor(Monitor):
    def check(self, deadline=None):
        self.drain()
        if deadline is not None and time.monotonic()>=deadline:
            raise TimeoutError('burst observation timed out')
        if not self.car.is_connected:
            raise ConnectionError('BLE disconnected')
        feedback=self.gps.latest
        pose=feedback.cars.get(self.car.number) if feedback and feedback.usable else None
        if pose is None:
            raise RuntimeError('UDP pose unavailable for measurement')
        return pose

    def wait(self, seconds, deadline, starting_heading=None):
        # The motion duration is bounded independently of camera delivery.
        # Record arriving UDP frames; assess measurement completeness afterwards.
        until=min(time.monotonic()+seconds,deadline)
        while time.monotonic()<until:
            self.drain()
            if not self.car.is_connected:
                raise ConnectionError('BLE disconnected')
            time.sleep(.02)

    def wait_stopped(self, seconds, deadline, starting_heading=None):
        until=time.monotonic()+seconds
        while time.monotonic()<until:
            self.drain()
            time.sleep(.03)
        return self.check(deadline)


def return_to_start(monitor, target, response):
    """Small single-axis corrections, each followed by a fresh UDP observation."""
    monitor.phase='return'
    for _ in range(8):
        pose=monitor.check()
        dx,dy=target[0]-pose.x,target[1]-pose.y
        heading=angle_delta(target[2],pose.angle)
        if math.hypot(dx,dy)<.05 and abs(heading)<7:
            return
        theta=math.radians(pose.angle)
        body=(dx*math.cos(theta)+dy*math.sin(theta),
              -dx*math.sin(theta)+dy*math.cos(theta))
        axis=2 if abs(heading)>=7 else (0 if abs(body[0])>=abs(body[1]) else 1)
        error=heading if axis==2 else body[axis]
        gain=abs(response[axis][axis])
        seconds=.45 if axis==2 else .3
        # Very short yaw bursts below motor deadband do no useful work on
        # weak-yaw robots. Use a modest start level and fewer settled attempts.
        low=.15 if axis==2 and abs(heading)>15 else .1 if axis==2 else .05
        high=.15 if axis==2 else .2
        magnitude=min(high,max(low,.4*abs(error)/max(.01,gain)/seconds))
        command=[0.,0.,0.]
        command[axis]=math.copysign(magnitude,error)
        try:
            monitor.car.set_velocity(*command)
            monitor.wait(seconds,time.monotonic()+2)
        finally:
            monitor.car.stop()
        monitor.wait_stopped(.7,time.monotonic()+3)
    print('Return paused after 8 short correction bursts; current pose:',monitor.check(),flush=True)


def save(path, value):
    with path.open('x', encoding='utf-8') as destination:
        json.dump(value, destination, indent=2)


def configs_match(left, right):
    """GATT float32 and JSON decimal readbacks may differ in their last digits."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys()==right.keys() and all(configs_match(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)) and isinstance(right,(list,tuple)):
        return len(left)==len(right) and all(configs_match(a,b) for a,b in zip(left,right))
    if isinstance(left,(int,float)) and isinstance(right,(int,float)):
        return math.isfinite(left) and math.isfinite(right) and abs(left-right)<1e-6
    return left==right


def checkpoint_config(root, config):
    # An explicit audit sidecar repairs snapshots from firmware whose full
    # config GATT value was stale after the dedicated trim write. Original
    # measurements and snapshots are retained, never silently overwritten.
    sidecar=root/'readback_reconciliation.json'
    if sidecar.exists():
        config=dict(config)
        config['trim_matrix']=json.loads(sidecar.read_text())['verified_trim']
    return config


def stage_result(root, name):
    result=json.loads((root/name/'result.json').read_text())
    if name=='axes-check' and result.get('errors'):
        for path in sorted(root.glob('axes-recheck-*/result.json'),reverse=True):
            certified=json.loads(path.read_text())
            if certified.get('verified'):
                return certified
    return result


class VerificationRollback:
    """Cover placement/preflight failures as well as failed motion samples."""
    def __init__(self, car):
        self.car=car
        self.baseline=None
        self.full=False

    def __enter__(self):
        return self

    def rollback(self):
        if self.baseline is None:
            return
        self.car.stop()
        if self.full:
            self.car.set_robot_config(self.baseline)
        else:
            self.car.set_trim([self.baseline['trim_matrix'][i:i+3] for i in (0,3,6)])
        self.baseline=None

    def __exit__(self, kind, error, traceback):
        if kind is not None:
            self.rollback()


def body_response(start, end, frames, since):
    # Integrate each world-frame step in its midpoint body frame. Using only
    # the starting heading confuses intentional yaw with transverse motion.
    poses = [start] + [p for received, seq, p, usable in frames
                       if received >= since and usable and p is not None] + [end]
    forward = left = yaw = 0.
    for p, q in zip(poses, poses[1:]):
        dtheta = angle_delta(q.angle, p.angle)
        theta = math.radians(p.angle+dtheta/2)
        dx, dy = q.x-p.x, q.y-p.y
        forward += dx*math.cos(theta)+dy*math.sin(theta)
        left += -dx*math.sin(theta)+dy*math.cos(theta)
        yaw += dtheta
    return [forward, left, yaw]


def steady_yaw_rate(samples):
    """Least-squares deg/s from timestamped headings, including angle unwrap."""
    if len(samples)<4 or samples[-1][0]-samples[0][0]<.4:
        return None
    unwrapped=[samples[0][1]]
    for previous,current in zip(samples,samples[1:]):
        unwrapped.append(unwrapped[-1]+angle_delta(current[1],previous[1]))
    average_t=sum(t for t,a in samples)/len(samples)
    average_a=sum(unwrapped)/len(unwrapped)
    denominator=sum((t-average_t)**2 for t,a in samples)
    return sum((t-average_t)*(a-average_a) for (t,raw),a in zip(samples,unwrapped))/denominator


def pulse(monitor, command, seconds, destination, pair=None):
    deadline = time.monotonic()+12.
    monitor.phase = 'baseline'
    start = monitor.wait_stopped(.4, deadline)
    monitor.phase = 'pulse_%s' % (command,)
    started = time.monotonic()
    try:
        monitor.car.set_velocity(*command)
        monitor.wait(seconds, deadline, start.angle)
    finally:
        monitor.car.stop()
        stopped = time.monotonic()
        monitor.velocity = (0., 0.)
        monitor.previous = None
    monitor.phase = 'settle'
    end = monitor.wait_stopped(1., deadline, start.angle)
    response = body_response(start, end, monitor.frames, started)
    frames = len({seq for received, seq, p, usable in monitor.frames
                  if started <= received <= stopped and p is not None and usable})
    if frames < 3:
        raise RuntimeError('too few fresh frames during pulse')
    result = dict(command=list(command), response=response, frames=frames,
                  duration_s=stopped-started, pair=pair,
                  requested_duration_s=seconds,
                  start=asdict(start), end=asdict(end))
    yaw_samples=[(received-started,p.angle) for received,seq,p,usable in monitor.frames
                 if started+.4<=received<=stopped and usable and p is not None]
    result['steady_yaw_deg_s']=steady_yaw_rate(yaw_samples)
    result['yaw_samples']=yaw_samples
    save(destination, result)
    print('command', command, 'response', [round(v, 4) for v in response], flush=True)
    return result


def burst_durations(scale):
    if not math.isfinite(scale) or not .5<=scale<=2:
        raise ValueError('burst scale must be between 0.5 and 2')
    return dict(forward=.8*scale,strafe=1.2*scale,yaw=1.2*scale,mixed=.7*scale)


def single_rows(rows):
    return [dict(axis=next(i for i,v in enumerate(r['command']) if v),
                 command_x=r['command'][0], command_y=r['command'][1], command_w=r['command'][2],
                 command=next(v for v in r['command'] if v), forward_m=r['response'][0],
                 left_m=r['response'][1], heading_deg=r['response'][2],
                 frames=r['frames'], duration_s=r['duration_s']) for r in rows]


def axis_errors(rows):
    errors = []
    for row in rows:
        axis = next(i for i,v in enumerate(row['command']) if v)
        response = row['response']
        if abs(response[axis]) < (5. if axis == 2 else .05):
            errors.append('insufficient response on axis %d' % axis)
        if response[axis]*row['command'][axis] <= 0:
            errors.append('wrong response sign on axis %d' % axis)
        for other in range(3):
            if other != axis and abs(response[other]) > (4. if other == 2 else .02):
                errors.append('axis %d drifts on axis %d' % (axis, other))
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--car', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('axes', 'axes-check', 'mixed', 'mixed-check'), required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--burst-scale',type=float,
                        help='multiply measurement durations (new runs default to 1.6)')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--step',type=float,default=1.,
                        help='fraction of the proposed linear trim update to apply (0,1]')
    parser.add_argument('--yaw-steps',type=float,nargs=2,
                        help='separate update fractions for forward/strafe yaw coupling')
    parser.add_argument('--mixed-step',type=float,default=1.,
                        help='fraction of fitted mixed-axis correction to apply (0,1]')
    parser.add_argument('--resume', action='store_true',
                        help='resume an interrupted stage with matching configuration')
    parser.add_argument('--retry', action='store_true',
                        help='record another verification attempt without overwriting earlier data')
    args = parser.parse_args(argv)
    if args.burst_scale is not None:
        burst_durations(args.burst_scale)
    if not math.isfinite(args.step) or not 0<args.step<=1:
        parser.error('step must be in (0,1]')
    if args.yaw_steps and any(not math.isfinite(v) or not 0<v<=1 for v in args.yaw_steps):
        parser.error('yaw steps must be in (0,1]')
    if not math.isfinite(args.mixed_step) or not 0<args.mixed_step<=1:
        parser.error('mixed step must be in (0,1]')
    if not args.execute:
        print('Preview: stage', args.stage, 'on car', args.car,
              '; --execute enables pulses, --apply enables candidate writes')
        return
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    stage = root / args.stage
    if args.retry:
        if args.stage not in ('axes-check','mixed-check') or not stage.is_dir():
            parser.error('retry requires an existing verification stage')
        stage=root/(args.stage+'-'+str(time.time_ns()))
    if args.resume:
        if not stage.is_dir() or (stage/'result.json').exists():
            parser.error('resume requires an interrupted stage')
    else:
        stage.mkdir(exist_ok=False)
    plan_path=stage/'plan.json'
    if args.resume:
        scale=json.loads(plan_path.read_text())['burst_scale'] if plan_path.exists() else 1.
        if args.burst_scale is not None and abs(args.burst_scale-scale)>1e-9:
            parser.error('resume must use the same burst scale as saved measurements')
    else:
        scale=1.6 if args.burst_scale is None else args.burst_scale
        save(plan_path,dict(burst_scale=scale,durations=burst_durations(scale)))
    durations=burst_durations(scale)
    gps = None
    # BLE first, then UDP, as in the now-tested diagonal tool.
    try:
        with Car(args.car, timeout=50) as car, VerificationRollback(car) as transaction:
            car.stop()
            config = car.read_robot_config()
            if args.resume:
                if not configs_match(config,checkpoint_config(root,json.loads((stage/'before.json').read_text()))):
                    raise RuntimeError('cannot resume under a different robot configuration')
            else:
                save(stage/'before.json', config)
            # Directional gains may be calibrated to cancel sign-independent
            # yaw bias. They remain fixed throughout each stage; full config
            # comparisons below prevent combining samples from different gains.
            prior = None
            if args.stage != 'axes':
                prior_name = {'axes-check':'axes', 'mixed':'axes-check', 'mixed-check':'mixed'}[args.stage]
                prior = stage_result(root,prior_name)
                if prior.get('errors') or (args.stage in ('axes-check','mixed-check') and not prior.get('applied')):
                    raise RuntimeError('preceding stage did not pass/apply')
                expected = checkpoint_config(root,prior['configuration'])
                if not configs_match(config,expected):
                    raise RuntimeError('configuration differs from preceding stage; do not mix calibration data')
                if args.stage in ('axes-check','mixed-check'):
                    source='axes' if args.stage=='axes-check' else 'mixed'
                    transaction.baseline=json.loads((root/source/'before.json').read_text())
                    transaction.full=args.stage=='mixed-check'
                    if transaction.full:
                        transaction.baseline=checkpoint_config(root,transaction.baseline)
            if args.stage in ('axes','axes-check','mixed') and any(config.get('mixed_trim', [0.]*18)):
                raise RuntimeError('mixed compensation must be zero before fitting a new baseline')
            gps = GPS(stale_after=.8)
            pose = car.wait_for_position(10)
            origin_path=stage/'origin.json'
            if args.resume:
                start_pose=json.loads(origin_path.read_text()) if origin_path.exists() else asdict(pose)
            else:
                start_pose=asdict(pose)
                save(origin_path,start_pose)
            origin=(start_pose['x'],start_pose['y'],start_pose['angle'])
            rows = ([json.loads(path.read_text()) for path in sorted(stage.glob('pulse_*.json'))]
                    if args.resume else [])
            saved_count=len(rows)
            errors = []
            frames_path=stage/('frames_resume_%d.csv'%time.time_ns() if args.resume else 'frames.csv')
            with frames_path.open('x',newline='',encoding='utf-8') as frames:
                writer = csv.writer(frames)
                writer.writerow(('elapsed_s','phase','frame_sequence','frame_age_ms','usable','x_m','y_m','heading_deg'))
                monitor = StageMonitor(car,gps,writer,SimpleQueue(),0,center=origin)
                try:
                    monitor.wait_stopped(.5, None)
                    if args.stage.startswith('axes'):
                        planned_index=0
                        for axis in range(3):
                            rows_before_axis=len(rows)
                            for sign in (1,-1):
                                command = [0.,0.,0.]
                                command[axis] = sign*(.1 if axis==2 else .5)
                                duration = durations[('forward','strafe','yaw')[axis]]
                                if planned_index<saved_count:
                                    if rows[planned_index]['command']!=command:
                                        raise RuntimeError('resume samples do not match pulse plan')
                                    planned_index+=1
                                    continue
                                planned_index+=1
                                rows.append(pulse(monitor,command,duration,stage/('pulse_%02d.json'%len(rows))))
                            response_model=[[.6,0.,0.],[0.,.6,0.],[0.,0.,140.]]
                            for output in range(3):
                                gains=[abs(r['response'][output]/r['duration_s']/r['command'][output])
                                       for r in rows if r['command'][output]]
                                if gains:
                                    response_model[output][output]=sum(gains)/len(gains)
                            if len(rows)>rows_before_axis:
                                return_to_start(monitor,origin,response_model)
                    else:
                        planned_index=0
                        for level in (.25,.5):
                            for pair_index,(a,b,third) in enumerate(PAIRS):
                                for sa,sb in ((1,1),(-1,-1),(1,-1),(-1,1)):
                                    command=[0.,0.,0.]
                                    command[a]=sa*level
                                    command[b]=sb*(.1 if b==2 else level)
                                    if planned_index<saved_count:
                                        if rows[planned_index]['command']!=command:
                                            raise RuntimeError('resume samples do not match pulse plan')
                                        planned_index+=1
                                        continue
                                    planned_index+=1
                                    rows.append(pulse(monitor,command,durations['mixed'],stage/('pulse_%02d.json'%len(rows)),pair_index))
                                    if len(rows)%2==0:
                                        response_model=(prior['response_matrix'] if args.stage=='mixed' else
                                            stage_result(root,'axes-check')['response_matrix'])
                                        return_to_start(monitor,origin,response_model)
                except BaseException as error:
                    car.stop()
                    # Preserve the exact trial configuration and completed
                    # samples for --resume. Only a completed, failed quality
                    # verification rolls back a candidate.
                    transaction.baseline=None
                    failure_path=stage/('failure_resume_%d.json'%time.time_ns() if args.resume else 'failure.json')
                    save(failure_path,dict(error=str(error),stage=args.stage))
                    raise
                finally:
                    car.stop()
                    monitor.close()
            candidate = None
            applied = False
            response = None
            if args.stage == 'axes':
                candidate=fit(single_rows(rows),car.read_trim())
                undamped=[list(row) for row in candidate['matrix']]
                candidate['matrix']=[list(row) for row in undamped]
                if args.step<1:
                    current=car.read_trim()
                    candidate['undamped_matrix']=candidate['matrix']
                    candidate['matrix']=[[current[i][j]+args.step*(candidate['matrix'][i][j]-current[i][j])
                                          for j in range(3)] for i in range(3)]
                    candidate['step']=args.step
                if args.yaw_steps:
                    current=car.read_trim()
                    for column,step in enumerate(args.yaw_steps):
                        candidate['matrix'][2][column]=current[2][column]+step*(undamped[2][column]-current[2][column])
                    candidate['yaw_steps']=args.yaw_steps
                errors=assess(single_rows(rows),candidate,car.read_trim())
                if not errors and args.apply:
                    car.set_trim(candidate['matrix'])
                    applied=True
            elif args.stage == 'axes-check':
                errors=axis_errors(rows)
                response=fit(single_rows(rows),car.read_trim())['response_matrix']
                if errors:
                    transaction.rollback()
            elif args.stage == 'mixed':
                candidate=fit_mixed(rows,prior['response_matrix'],deadbands=(.015,.015,3.),
                                    step=args.mixed_step)
                if args.apply:
                    config['mixed_trim']=candidate['mixed_trim']
                    car.set_robot_config(config)
                    applied=True
            else:
                baseline_rows=[json.loads(p.read_text()) for p in sorted((root/'mixed').glob('pulse_*.json'))]
                for row, baseline_row in zip(rows,baseline_rows):
                    third=PAIRS[row['pair']][2]
                    limit=3. if third==2 else .015
                    value=abs(row['response'][third])
                    if value>limit:
                        errors.append('residual failed on command %s'%row['command'])
                if errors:
                    config['mixed_trim']=json.loads((root/'mixed'/'before.json').read_text())['mixed_trim']
                    transaction.rollback()
            # Firmware writes floats for axis trim; fetch canonical readback.
            if (applied and args.stage=='mixed') or (args.stage=='mixed-check' and errors):
                # Full-config writes reboot. Persist the accepted candidate;
                # the next stage independently reconnects and checks readback.
                final_config=config
            else:
                final_config=car.read_robot_config()
            result=dict(car=args.car,stage=args.stage,errors=errors,applied=applied,
                        proposal=candidate,response_matrix=response,configuration=final_config)
            save(stage/'result.json',result)
            print('Stage',args.stage,'errors:',errors,'applied:',applied,flush=True)
            if errors:
                raise SystemExit(1)
    finally:
        if gps is not None:
            gps.close()


if __name__=='__main__':
    main()
