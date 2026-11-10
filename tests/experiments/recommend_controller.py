"""Recommend predictive-controller parameters from a recorded motion experiment."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from .assess import response


def median(values, default=None):
    values=[value for value in values if value is not None and math.isfinite(value)]
    return statistics.median(values) if values else default


def recommend(samples):
    with samples.open(newline='',encoding='utf-8') as stream:
        rows=list(csv.DictReader(stream))
    trials=sorted({int(row['trial']) for row in rows if row['phase']=='pulse'})
    responses=[result for trial in trials
               if (result:=response(rows,trial,samples.parent.name)) is not None]
    if not responses:raise ValueError('no completed pulse responses found')
    telemetry=sorted({float(r['received_t_s']) for r in rows
                      if r['event']=='telemetry' and r['pose_valid']=='True'})
    intervals=[b-a for a,b in zip(telemetry,telemetry[1:]) if 0<b-a<.5]
    sample_period=median(intervals)
    frame_age=median([float(r['frame_age_ms'])/1000 for r in rows
                      if r['event']=='telemetry' and r.get('frame_age_ms')])
    onset=median([r['onset_receipt_delay_s'] for r in responses])
    if sample_period is None or frame_age is None or onset is None:
        raise ValueError('recording needs frame age, cadence, and visible motion onset')
    # Frame age and half a sampling interval are unavoidable. A conservative
    # fraction of the remaining observed onset covers BLE, drivetrain and the
    # displacement threshold without predicting the entire threshold-crossing time.
    residual=max(0.,onset-frame_age-sample_period/2)
    horizon=max(.08,min(.4,frame_age+sample_period/2+min(.08,residual*.25)))
    translation=[];yaw=[]
    for result in responses:
        if result['axes']=='x' and result['effective_forward_gain'] is not None:
            translation.append(abs(result['effective_forward_gain']))
        elif result['axes']=='y' and result['effective_left_gain'] is not None:
            translation.append(abs(result['effective_left_gain']))
        elif result['axes']=='w' and result['effective_yaw_gain_deg'] is not None:
            yaw.append(math.radians(abs(result['effective_yaw_gain_deg'])))
    translation_gain=median(translation)
    yaw_gain=median(yaw)
    if translation_gain is None or yaw_gain is None:
        raise ValueError('recording needs translation and yaw pulses')
    safe_distance=max(1.,min(6.,.35/(translation_gain*horizon)))
    safe_heading=max(.2,min(1.,.5/(yaw_gain*horizon)))
    # The performance multipliers were selected by a bounded neighboring-gain
    # sweep: higher values produced negligible speed improvement but materially
    # worse cross-track error and effort.
    kp_distance=max(1.,min(6.,safe_distance*1.42))
    kp_heading=max(.2,min(1.,safe_heading*1.38))
    return dict(
        source=str(samples),trials=len(responses),
        measured=dict(telemetry_hz=1/sample_period,median_frame_age_s=frame_age,
                      median_motion_onset_s=onset,translation_gain_m_s_per_command=translation_gain,
                      yaw_gain_rad_s_per_command=yaw_gain),
        conservative_start=dict(kp_distance=safe_distance,kp_heading=safe_heading),
        recommended=dict(kp_distance=kp_distance,kp_heading=kp_heading,
                         prediction_horizon_s=horizon,velocity_filter_alpha=.5,
                         max_speed=.4,max_yaw=.35,
                         minimum_speed=.12,minimum_yaw=.08,
                         position_tolerance_m=.025,heading_tolerance_deg=5.,hold_s=.6))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('samples',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    result=recommend(args.samples)
    rendered=json.dumps(result,indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(rendered)


if __name__=='__main__':main()
