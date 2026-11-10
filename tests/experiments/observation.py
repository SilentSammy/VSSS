"""Passive preflight recording: this module never imports or constructs Car."""
import json
from threading import Lock
import time
from .recording import Recorder


def observe(args, directory, gps, clock=time.monotonic, sleep=time.sleep):
    recorder = Recorder(directory, None, gps, (args.center_x,args.center_y,args.heading),
                        clock=clock, car_number=args.car)
    receipts=[]
    receipt_lock=Lock()
    def received(feedback):
        with receipt_lock:
            receipts.append(feedback)
    gps.add_listener(received)
    started=clock()
    end=clock()+args.observe
    try:
        recorder.set_phase('observe',0)
        while clock()<end and not (directory/'STOP').exists():
            recorder.record('observe')
            sleep(args.period)
    finally:
        ended=clock()
        gps.remove_listener(received)
        recorder.close()
    with receipt_lock:
        packets=[p for p in receipts if started <= p.received_at <= ended]
    gaps=[b.received_at-a.received_at for a,b in zip(packets,packets[1:])]
    trailing=None if not packets else max(0.,ended-packets[-1].received_at)
    max_gap=max(gaps+[trailing]) if packets else None
    recent_unicast=getattr(gps,'_unicast_received',None)
    source=getattr(gps,'_server',None)
    transport=('unicast' if args.telemetry=='vsss' and source is not None and recent_unicast is not None
               and ended-recent_unicast < 2 else 'broadcast') if packets else 'none'
    first_marker=next((i for i,p in enumerate(packets) if p.usable and args.car in p.cars),None)
    missing_marker_packets=0 if first_marker is None else sum(
        p.usable and args.car not in p.cars for p in packets[first_marker:])
    unusable_packets=sum(not p.usable for p in packets)
    visible=gps.get_car(args.car) is not None
    ready=(visible and max_gap is not None and max_gap <= gps.stale_after
           and missing_marker_packets==0 and unusable_packets==0)
    reason = None
    if not ready:
        if not packets:
            reason = gps.feedback_problem or 'No telemetry packets received.'
        elif not visible:
            reason = gps.feedback_problem or f'Marker {args.car} is not currently visible.'
        elif max_gap > gps.stale_after:
            reason = f'A {max_gap:.3f} s delivery gap exceeded the {gps.stale_after:.3f} s freshness limit.'
        elif unusable_packets:
            reason = f'{unusable_packets} packet(s) were delayed, stale on arrival, or out of order.'
        else:
            reason = f'Marker {args.car} disappeared in {missing_marker_packets} packet(s).'
    excesses=[p.network_excess_delay_s for p in packets if p.network_excess_delay_s is not None]
    result=dict(status='observed',command_owned=False,ready=ready,readiness_reason=reason,
                telemetry_packets=len(packets),usable_packets=len(packets)-unusable_packets,
                unusable_packets=unusable_packets,missing_marker_packets=missing_marker_packets,
                max_network_excess_delay_s=max(excesses,default=None),balls=len(gps.balls),transport=transport,
                source=source if transport=='unicast' else None,
                first_packet_delay_s=None if not packets else packets[0].received_at-started,
                seconds_since_last_packet=trailing,max_receipt_gap_s=max_gap,
                freshness_limit_s=gps.stale_after,feedback_problem=gps.feedback_problem)
    for name in ('result.json','status.json'):
        (directory/name).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    return 0 if result['ready'] else 2
