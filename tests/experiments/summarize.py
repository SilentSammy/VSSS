"""Descriptive pulse responses, not an automatically fitted dynamics model."""
import argparse
import csv
import glob
import math
from pathlib import Path


def slope(times, values):
    if len(times) < 3:
        return None
    mt, mv = sum(times)/len(times), sum(values)/len(values)
    denominator = sum((t-mt)**2 for t in times)
    return None if denominator == 0 else sum((t-mt)*(v-mv) for t,v in zip(times,values))/denominator


def unwrap(angles):
    result = [angles[0]]
    for previous, current in zip(angles, angles[1:]):
        result.append(result[-1]+(current-previous+180)%360-180)
    return result


def summarize(path):
    with path.open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    results = []
    trials = sorted({(int(row['trial']),int(row.get('segment',0))) for row in rows
                     if row['phase']=='pulse' and row['event']=='control'})
    for trial,segment in trials:
        pulse = [r for r in rows if r['phase']=='pulse' and int(r['trial'])==trial
                 and int(r.get('segment',0))==segment]
        command = next((r for r in pulse if r['event']=='control' and any(float(r[k])!=0 for k in ('input_x','input_y','input_w'))), None)
        frames = [r for r in pulse if r['event']=='telemetry' and r['pose_valid']=='True'
                  and (command is None or float(r['t_s'])>=float(command['t_s']))]
        if command is None or not frames:
            continue
        start = float(command['t_s'])
        prior = [r for r in rows if r['event']=='telemetry' and r['pose_valid']=='True' and float(r['t_s'])<start]
        baseline = prior[-1] if prior else frames[0]
        last_segment = max(s for t,s in trials if t==trial)
        settled = [r for r in rows if int(r['trial'])==trial and r['phase']=='settle' and r['event']=='telemetry' and r['pose_valid']=='True'] if segment==last_segment else []
        final = settled[-1] if settled else frames[-1]
        theta = math.radians(float(baseline['heading_deg']))
        times = [float(r['received_t_s']) for r in frames]
        xs, ys = [float(r['x_m']) for r in frames], [float(r['y_m']) for r in frames]
        headings = unwrap([float(r['heading_deg']) for r in frames])
        vx, vy, omega = slope(times,xs), slope(times,ys), slope(times,headings)
        dx,dy = float(final['x_m'])-float(baseline['x_m']), float(final['y_m'])-float(baseline['y_m'])
        trajectory = [r for r in rows if r['event']=='telemetry' and r['pose_valid']=='True'
                      and float(baseline['t_s']) <= float(r['t_s']) <= float(final['t_s'])]
        full_headings = unwrap([float(r['heading_deg']) for r in trajectory])
        shortened = any(r['event']=='boundary_recenter' and int(r['trial'])==trial
                        and int(r.get('segment',0))==segment for r in rows)
        results.append(dict(run=path.parent.name, trial=trial, segment=segment,
                            trial_name=command.get('trial_name',''),
                            input_x=float(command['input_x']), input_y=float(command['input_y']), input_w=float(command['input_w']),
                            pose_samples=len(frames), boundary_shortened=shortened,
                            pulse_finished=not shortened and (bool(settled) or segment<last_segment),
                            forward_m_s=None if vx is None else vx*math.cos(theta)+vy*math.sin(theta),
                            left_m_s=None if vx is None else -vx*math.sin(theta)+vy*math.cos(theta),
                            rotation_deg_s=omega,
                            net_forward_m=dx*math.cos(theta)+dy*math.sin(theta),
                            net_left_m=-dx*math.sin(theta)+dy*math.cos(theta),
                            net_heading_deg=full_headings[-1]-full_headings[0]))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', help='CSV paths or glob patterns')
    parser.add_argument('--output',type=Path,default=Path('tests/experiments/runs/summary.csv'))
    args=parser.parse_args()
    paths=sorted({Path(p) for pattern in args.files for p in glob.glob(pattern)})
    results=[row for path in paths for row in summarize(path)]
    if not results:
        parser.error('No pulse data found')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(results[0]))
        writer.writeheader();writer.writerows(results)
    for row in results:
        print(row)
    print(f'Summary: {args.output.resolve()}')


if __name__=='__main__':
    main()
