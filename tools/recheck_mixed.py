"""Recheck above-tolerance axis or mixed cases, retaining passed data.

Uses the identical candidate and two fresh repeats. A case passes when the
median absolute third-axis displacement of the three observations meets the
original tolerance. Every repeat is saved. Rolls back on failed measurement.
"""
import argparse
import csv
import json
from pathlib import Path
from queue import SimpleQueue
import statistics
import time

from vsss import Car, GPS
from tools._fit_mixed import PAIRS
from tools._fit_trim import fit
from tools.calibrate_staged import (StageMonitor, VerificationRollback,
    checkpoint_config, configs_match, pulse, return_to_start, save, single_rows, axis_errors)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--car',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verification',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--axes',action='store_true',help='recheck single-axis verification instead')
    args=parser.parse_args(argv)
    if not args.execute:
        print('Preview: repeat only above-tolerance cases from',args.verification)
        return
    root=args.output
    fit_stage='axes' if args.axes else 'mixed'
    prior=json.loads((root/fit_stage/'result.json').read_text())
    candidate=checkpoint_config(root,prior['configuration'])
    baseline=checkpoint_config(root,json.loads((root/fit_stage/'before.json').read_text()))
    source=json.loads((args.verification/'before.json').read_text())
    if not configs_match(candidate,source):
        raise RuntimeError('source measurements do not match the candidate')
    rows=[json.loads(p.read_text()) for p in sorted(args.verification.glob('pulse_*.json'))]
    total=6 if args.axes else 24
    if len(rows)!=total:
        raise RuntimeError('recheck requires a complete verification')
    pending=[]
    for i,row in enumerate(rows):
        active=next(j for j,v in enumerate(row['command']) if v)
        tests=([j for j in range(3) if j!=active] if args.axes else [PAIRS[row['pair']][2]])
        failed=[]
        for third in tests:
            limit=(4. if third==2 else .02) if args.axes else (3. if third==2 else .015)
            if abs(row['response'][third])>limit:
                failed.append((third,limit))
        if failed:
            pending.append((i,row,failed))
    directory=root/(fit_stage+'-recheck-'+str(time.time_ns()))
    directory.mkdir()
    print('Rechecking',len(pending),'cases; retaining',total-len(pending),'passed cases',flush=True)
    car=None
    gps=None
    try:
        car=Car(args.car,timeout=50)
        car.stop()
        config=car.read_robot_config()
        if not (configs_match(config,baseline) or configs_match(config,candidate)):
            raise RuntimeError('robot configuration changed since verification')
        if not configs_match(config,candidate):
            car.set_robot_config(candidate)
            car.close()
            car=Car(args.car,timeout=50)
        with VerificationRollback(car) as transaction:
            transaction.baseline=baseline
            transaction.full=True
            actual=car.read_robot_config()
            if not configs_match(actual,candidate):
                raise RuntimeError('candidate readback mismatch')
            save(directory/'before.json',actual)
            gps=GPS(stale_after=.8)
            origin_pose=car.wait_for_position(10)
            origin=(origin_pose.x,origin_pose.y,origin_pose.angle)
            response=json.loads((root/'axes-check/result.json').read_text())['response_matrix']
            decisions=[]
            errors=[]
            with (directory/'frames.csv').open('x',newline='',encoding='utf-8') as recording:
                writer=csv.writer(recording)
                writer.writerow(('elapsed_s','phase','frame_sequence','frame_age_ms','usable','x_m','y_m','heading_deg'))
                monitor=StageMonitor(car,gps,writer,SimpleQueue(),0,center=origin)
                try:
                    for index,row,failed in pending:
                        observations=[row]
                        for repeat in range(2):
                            duration=(.8 if row['command'][0] else 1.2) if args.axes else .7
                            duration=row.get('requested_duration_s',duration)
                            measured=pulse(monitor,row['command'],duration,
                                directory/('repeat_%02d_%d.json'%(index,repeat)),row['pair'])
                            observations.append(measured)
                            return_to_start(monitor,origin,response)
                        for third,limit in failed:
                            values=[abs(o['response'][third]) for o in observations]
                            median=statistics.median(values)
                            decisions.append(dict(index=index,axis=third,command=row['command'],
                                absolute_residuals=values,median=median,limit=limit))
                            if median>limit:
                                errors.append('repeatable residual on %s'%row['command'])
                        if args.axes:
                            rows[index]=dict(row,response=[statistics.median(o['response'][j] for o in observations)
                                                         for j in range(3)],
                                             duration_s=statistics.median(o['duration_s'] for o in observations))
                finally:
                    car.stop()
                    monitor.close()
            if args.axes:
                errors=list(dict.fromkeys(errors+axis_errors(rows)))
            candidate_matrix=[candidate['trim_matrix'][i:i+3] for i in (0,3,6)]
            verified_response=fit(single_rows(rows),candidate_matrix)['response_matrix'] if args.axes else None
            if errors:
                transaction.rollback()
            save(directory/'result.json',dict(source=str(args.verification),
                retained_samples=total-len(pending),decisions=decisions,errors=errors,
                response_matrix=verified_response,applied=False,
                verified=not errors,configuration=baseline if errors else candidate))
            print('Recheck errors:',errors,'saved:',directory,flush=True)
            if errors:
                raise SystemExit(1)
    finally:
        if car is not None:
            car.close()
        if gps is not None:
            gps.close()


if __name__=='__main__':
    main()
