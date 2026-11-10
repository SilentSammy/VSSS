"""Score waypoint recordings with common position, heading, time and effort metrics."""
import argparse
import csv
import json
import math
from pathlib import Path

from .controllers import wrap_angle


def _number(row,key,default=0.):
    try:return float(row[key])
    except (KeyError,TypeError,ValueError):return default


def score(path):
    with path.open(newline='',encoding='utf-8') as stream:
        rows=list(csv.DictReader(stream))
    results=[]
    for trial in sorted({int(r['trial']) for r in rows if r['phase']=='waypoint'}):
        samples=[r for r in rows if r['phase']=='waypoint' and int(r['trial'])==trial
                 and r['event']=='telemetry' and r['pose_valid']=='True']
        controls=[r for r in rows if r['phase']=='waypoint' and int(r['trial'])==trial
                  and r['event']=='control']
        if len(samples)<2:continue
        start=samples[0]; target=samples[-1]
        sx,sy=_number(start,'x_m'),_number(start,'y_m')
        tx,ty=_number(target,'target_x_m'),_number(target,'target_y_m')
        th=math.radians(_number(target,'target_heading_deg'))
        ux,uy=tx-sx,ty-sy; length=math.hypot(ux,uy)
        if length:ux,uy=ux/length,uy/length
        errors=[]; heading_errors=[]; overshoots=[]; cross=[]
        for row in samples:
            x,y=_number(row,'x_m'),_number(row,'y_m')
            dx,dy=x-tx,y-ty
            errors.append(math.hypot(dx,dy))
            heading_errors.append(abs(math.degrees(wrap_angle(th-math.radians(_number(row,'heading_deg'))))))
            if length:
                along=(x-sx)*ux+(y-sy)*uy
                overshoots.append(max(0.,along-length))
                cross.append(abs((x-sx)*uy-(y-sy)*ux))
        times=[_number(r,'received_t_s') for r in samples]
        iae=sum((errors[i]+errors[i-1])*.5*(times[i]-times[i-1]) for i in range(1,len(times)))
        effort=[math.sqrt(sum(_number(r,key)**2 for key in ('input_x','input_y','input_w'))) for r in controls]
        results.append(dict(
            trial=trial, name=samples[-1].get('trial_name',''),
            duration_s=_number(samples[-1],'t_s')-_number(samples[0],'t_s'),
            final_position_error_m=errors[-1], final_heading_error_deg=heading_errors[-1],
            position_iae_m_s=iae, max_position_error_m=max(errors),
            overshoot_m=max(overshoots,default=0.), cross_track_m=max(cross,default=0.),
            rms_command=math.sqrt(sum(v*v for v in effort)/len(effort)) if effort else 0.,
            max_minimum_pose_age_s=max((_number(r,'minimum_pose_age_s') for r in samples),default=0.)))
    pauses=sum(r['event']=='telemetry_pause' for r in rows)
    if not results:raise ValueError('no complete waypoint telemetry found')
    aggregate=dict(
        trials=len(results), telemetry_pauses=pauses,
        mean_duration_s=sum(r['duration_s'] for r in results)/len(results),
        mean_final_position_error_m=sum(r['final_position_error_m'] for r in results)/len(results),
        mean_final_heading_error_deg=sum(r['final_heading_error_deg'] for r in results)/len(results),
        mean_position_iae_m_s=sum(r['position_iae_m_s'] for r in results)/len(results),
        max_overshoot_m=max(r['overshoot_m'] for r in results),
        max_cross_track_m=max(r['cross_track_m'] for r in results),
        mean_rms_command=sum(r['rms_command'] for r in results)/len(results))
    return dict(source=str(path),aggregate=aggregate,trials=results)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('samples',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=score(args.samples)
    output=args.output or args.samples.with_name('waypoint_metrics.json')
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result['aggregate'],indent=2))
    print(f'Metrics: {output.resolve()}')


if __name__=='__main__':main()
