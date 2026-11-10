"""Curate accepted trials and measure rest-to-rest responses without fitting a model."""
import argparse
import csv
import json
import math
from pathlib import Path

from .summarize import unwrap


INPUTS = ('input_x', 'input_y', 'input_w')


def response(rows, trial, run):
    selected = [r for r in rows if int(r['trial']) == trial]
    active = [r for r in selected if r['phase'] == 'pulse' and r['event'] == 'control'
              and any(float(r[k]) != 0 for k in INPUTS)]
    rests = [r for r in selected if r['phase'] == 'settle' and r['event'] == 'telemetry'
             and r['pose_valid'] == 'True']
    if not active or not rests:
        return None
    start = float(active[0]['t_s'])
    preceding = [r for r in rows if r['event'] == 'telemetry' and r['pose_valid'] == 'True'
                 and float(r['t_s']) < start]
    if not preceding:
        return None
    baseline, final = preceding[-1], rests[-1]
    first, last = float(baseline['t_s']), float(final['t_s'])
    trajectory = [r for r in rows if r['event'] == 'telemetry' and r['pose_valid'] == 'True'
                  and first <= float(r['t_s']) <= last]
    forward = left = rotation = 0.
    for p, q in zip(trajectory, trajectory[1:]):
        da = (float(q['heading_deg']) - float(p['heading_deg']) + 180) % 360 - 180
        theta = math.radians(float(p['heading_deg']) + da / 2)
        dx, dy = float(q['x_m']) - float(p['x_m']), float(q['y_m']) - float(p['y_m'])
        forward += dx * math.cos(theta) + dy * math.sin(theta)
        left += -dx * math.sin(theta) + dy * math.cos(theta)
        rotation += da
    # Integrate held command from its queue time through the end of the rest.
    events = [r for r in selected if start <= float(r['t_s']) <= last]
    exposure = [0., 0., 0.]
    for p, q in zip(events, events[1:]):
        dt = float(q['t_s']) - float(p['t_s'])
        for i, key in enumerate(INPUTS):
            exposure[i] += float(p[key]) * dt
    for i, key in enumerate(INPUTS):
        exposure[i] += float(events[-1][key]) * (last - float(events[-1]['t_s']))
    axes = ''.join(axis for axis, key in zip('xyw', INPUTS) if any(float(r[key]) != 0 for r in active))
    # Onset is a receipt-domain threshold crossing, not a BLE propagation time.
    onset = None
    frames = [r for r in trajectory if float(r['t_s']) >= start]
    for r in frames:
        distance = math.hypot(float(r['x_m'])-float(baseline['x_m']),
                              float(r['y_m'])-float(baseline['y_m']))
        angle = abs((float(r['heading_deg'])-float(baseline['heading_deg'])+180)%360-180)
        if (axes == 'w' and angle >= 3) or (axes != 'w' and distance >= .005):
            onset = float(r['received_t_s']) - start
            break
    stop = next((r for r in selected if r['phase']=='settle' and r['event']=='stop_request'), None)
    stop_ack = next((r for r in selected if r['phase']=='settle' and r['event']=='stop_ack'), None)
    stopped = None
    if stop:
        stop_time = float(stop['t_s'])
        for i, r in enumerate(rests):
            tail = rests[i:]
            if float(tail[-1]['received_t_s'])-float(r['received_t_s']) < .3:
                continue
            translation_span = math.hypot(
                max(float(p['x_m']) for p in tail)-min(float(p['x_m']) for p in tail),
                max(float(p['y_m']) for p in tail)-min(float(p['y_m']) for p in tail))
            angles = unwrap([float(p['heading_deg']) for p in tail])
            if translation_span <= .006 and max(angles)-min(angles) <= 2:
                stopped = float(r['received_t_s'])-stop_time
                break
    gains = [value/u if abs(u) > .001 else None
             for value,u in zip((forward,left,rotation),exposure)]
    gaps = [float(q['received_t_s'])-float(p['received_t_s']) for p,q in zip(trajectory,trajectory[1:])]
    return dict(run=run, trial=trial, name=active[0].get('trial_name') or axes, axes=axes,
                pose_samples=len(trajectory), forward_m=forward, left_m=left, heading_change_deg=rotation,
                x_command_area=exposure[0], y_command_area=exposure[1], w_command_area=exposure[2],
                effective_forward_gain=gains[0], effective_left_gain=gains[1], effective_yaw_gain_deg=gains[2],
                onset_receipt_delay_s=onset, stationary_receipt_delay_s=stopped,
                stop_ack_delay_s=None if not stop or not stop_ack else float(stop_ack['t_s'])-float(stop['t_s']),
                max_telemetry_gap_s=max(gaps,default=0))


def assess(selection):
    specification = json.loads(selection.read_text())
    events, responses, fields = [], [], ['source_run']
    for source in specification['sources']:
        path = (selection.parent / source['csv']).resolve()
        with path.open(newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            fields.extend(key for key in reader.fieldnames if key not in fields)
            rows = [row for row in reader if int(row['trial']) in source['include_trials']]
        run = path.parent.name
        events.extend(dict(source_run=run, **r) for r in rows)
        for trial in source['include_trials']:
            result = response(rows, trial, run)
            if result is not None:
                responses.append(result)
    directory = selection.parent
    with (directory/'accepted_events.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=fields)
        writer.writeheader();writer.writerows(events)
    if responses:
        with (directory/'responses.csv').open('w',newline='',encoding='utf-8') as file:
            writer=csv.DictWriter(file,fieldnames=list(responses[0]))
            writer.writeheader();writer.writerows(responses)
    print(f'Accepted {len(events)} events, {len(responses)} completed strides.')
    for result in responses:
        print(result)
    return responses


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('selection',type=Path)
    assess(parser.parse_args().selection)
