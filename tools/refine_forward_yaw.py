"""Refine one translation-to-yaw coefficient using measured secant steps.

All other axis/mixed coefficients are preserved. Completed pulses and trials
are reused by --resume. Motion uses BLE bursts and host UDP only.
"""
import argparse
import csv
import json
from pathlib import Path
from queue import SimpleQueue
import time

from vsss import Car,GPS
from tools.calibrate_staged import (StageMonitor,configs_match,pulse,return_to_start,save)


def secant_coefficient(c0,q0,c1,q1,max_change=.06):
    if abs(c1-c0)<1e-9 or abs(q1-q0)<.5:
        raise ValueError('probe did not establish a measurable yaw sensitivity')
    change=-q0*(c1-c0)/(q1-q0)
    return max(-2.,min(2.,c0+max(-max_change,min(max_change,change))))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--car',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--response',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=1.28)
    parser.add_argument('--axis',type=int,choices=(0,1),default=0,
                        help='translation axis: 0 forward, 1 strafe')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    if not .8<=args.seconds<=2.5:
        parser.error('duration must be 0.8-2.5 seconds')
    if not args.execute:
        print('Preview: axis',args.axis,'bursts of',args.seconds,'seconds; one trim coefficient changes')
        return
    root=args.output
    root.mkdir(parents=True,exist_ok=args.resume)
    gps=None
    with Car(args.car,timeout=50) as car:
        car.stop()
        config=car.read_robot_config()
        if args.resume:
            baseline=json.loads((root/'before.json').read_text())
            comparable=dict(config,trim_matrix=list(config['trim_matrix']))
            comparable['trim_matrix'][6+args.axis]=baseline['trim_matrix'][6+args.axis]
            if not configs_match(comparable,baseline):
                raise RuntimeError('configuration changed beyond selected translation/yaw coefficient')
            plan=json.loads((root/'plan.json').read_text())
            if abs(plan['seconds']-args.seconds)>1e-9:
                raise RuntimeError('resume duration differs')
            if plan.get('axis',0)!=args.axis:
                raise RuntimeError('resume axis differs')
        else:
            baseline=config
            save(root/'before.json',baseline)
            save(root/'plan.json',dict(seconds=args.seconds,car=args.car,axis=args.axis))
        response=json.loads(args.response.read_text())['response_matrix']
        try:
            gps=GPS(stale_after=.8)
            p=car.wait_for_position(10)
            origin_path=root/'origin.json'
            if origin_path.exists():
                origin=tuple(json.loads(origin_path.read_text()))
            else:
                origin=(p.x,p.y,p.angle)
                save(origin_path,origin)
            with (root/('frames_%d.csv'%time.time_ns())).open('x',newline='',encoding='utf-8') as recording:
                writer=csv.writer(recording)
                writer.writerow(('elapsed_s','phase','frame_sequence','frame_age_ms','usable','x_m','y_m','heading_deg'))
                monitor=StageMonitor(car,gps,writer,SimpleQueue(),0,center=origin)
                def trial(index,coefficient):
                    directory=root/('trial_%02d'%index)
                    directory.mkdir(exist_ok=True)
                    result_path=directory/'result.json'
                    if result_path.exists():
                        result=json.loads(result_path.read_text())
                        if abs(result['coefficient']-coefficient)>1e-6:
                            raise RuntimeError('trial coefficient differs from checkpoint')
                        return result
                    matrix=[baseline['trim_matrix'][i:i+3] for i in (0,3,6)]
                    matrix[2][args.axis]=coefficient
                    car.set_trim(matrix)
                    monitor.wait_stopped(.5,None)
                    rows=[]
                    for j,sign in enumerate((1,-1)):
                        path=directory/('pulse_%d.json'%j)
                        command=[0.,0.,0.]
                        command[args.axis]=sign*.5
                        row=json.loads(path.read_text()) if path.exists() else pulse(
                            monitor,command,args.seconds,path)
                        rows.append(row)
                    return_to_start(monitor,origin,response)
                    rates=[r.get('steady_yaw_deg_s') for r in rows]
                    if any(rate is None for rate in rates):
                        raise RuntimeError('not enough steady yaw frames; resume with saved measurements')
                    result=dict(coefficient=coefficient,rates=rates,
                                net_yaw=[r['response'][2] for r in rows],
                                signed_gain=rates[0]-rates[1],score=max(abs(v) for v in rates))
                    save(result_path,result)
                    print('trial',index,json.dumps(result),flush=True)
                    return result
                try:
                    initial=baseline['trim_matrix'][6+args.axis]
                    first=trial(0,initial)
                    probe=trial(1,min(1.9,initial+.02))
                    target=secant_coefficient(initial,first['signed_gain'],
                                               probe['coefficient'],probe['signed_gain'])
                    fitted=trial(2,target)
                    best=min((first,probe,fitted),key=lambda r:r['score'])
                    verified=trial(3,best['coefficient'])
                    accepted=verified['score']<=2. and max(abs(v) for v in verified['net_yaw'])<=4.
                    final=best['coefficient'] if accepted else initial
                    matrix=[baseline['trim_matrix'][i:i+3] for i in (0,3,6)]
                    matrix[2][args.axis]=final
                    actual=car.set_trim(matrix)
                    save(root/'result.json',dict(accepted=accepted,before=first,
                        selected=best,verification=verified,trim=actual,
                        note='mixed coefficients preserved; require mixed regression after linear update'))
                    print('Accepted:',accepted,'saved:',root,flush=True)
                finally:
                    car.stop()
                    monitor.close()
        finally:
            if gps is not None:
                gps.close()


if __name__=='__main__':
    main()
