"""Measure VSSS telemetry timing from an ephemeral subscriber socket."""
import argparse
import json
import math
from pathlib import Path
import socket
import statistics
import time


def percentile(values, fraction):
    if not values:return None
    ordered=sorted(values)
    position=(len(ordered)-1)*fraction
    lower=math.floor(position);upper=math.ceil(position)
    return ordered[lower] if lower==upper else ordered[lower]+(ordered[upper]-ordered[lower])*(position-lower)


def benchmark(duration, host='255.255.255.255', control_port=45889, marker=1):
    sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET,socket.SO_BROADCAST,1)
    sock.bind(('',0));sock.settimeout(.1)
    started=time.monotonic();deadline=started+duration;next_subscription=started
    arrivals=[];packets=[];source=None
    try:
        while time.monotonic()<deadline:
            now=time.monotonic()
            if now>=next_subscription:
                sock.sendto(b'VSSS_SUBSCRIBE',(host,control_port))
                next_subscription=now+1
            try:data,address=sock.recvfrom(65535)
            except socket.timeout:continue
            try:packet=json.loads(data)
            except (UnicodeError,ValueError):continue
            if not isinstance(packet,dict) or 'markers' not in packet:continue
            source=address
            arrivals.append(time.monotonic());packets.append(packet)
    finally:
        if source:
            try:sock.sendto(b'VSSS_UNSUBSCRIBE',(source[0],control_port))
            except OSError:pass
        sock.close()
    gaps=[b-a for a,b in zip(arrivals,arrivals[1:])]
    ages=[float(p['frame_age_ms']) for p in packets if p.get('frame_age_ms') is not None]
    stamps=[float(p['timestamp']) for p in packets if p.get('timestamp') is not None]
    sequences=[int(p['frame_sequence']) for p in packets if p.get('frame_sequence') is not None]
    sequence_steps=[b-a for a,b in zip(sequences,sequences[1:]) if b>=a]
    server_span=stamps[-1]-stamps[0] if len(stamps)>1 else None
    capture_fps=((sequences[-1]-sequences[0])/server_span
                 if len(sequences)>1 and server_span and server_span>0 else None)
    return dict(
        duration_s=duration,source=source,packets=len(packets),
        receive_hz=((len(arrivals)-1)/(arrivals[-1]-arrivals[0]) if len(arrivals)>1 else 0.),
        receipt_gap_ms=dict(median=1000*statistics.median(gaps) if gaps else None,
                            p95=1000*percentile(gaps,.95) if gaps else None,
                            maximum=1000*max(gaps) if gaps else None),
        frame_age_ms=dict(median=statistics.median(ages) if ages else None,
                          p95=percentile(ages,.95),maximum=max(ages) if ages else None),
        capture_fps=capture_fps,
        sequence_step=dict(median=statistics.median(sequence_steps) if sequence_steps else None,
                           p95=percentile(sequence_steps,.95),maximum=max(sequence_steps) if sequence_steps else None),
        marker_packets=sum(any(int(item.get('id',-1))==marker for item in p['markers']) for p in packets),
        ball_packets=sum(bool(p.get('balls')) for p in packets))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration',type=float,default=15.)
    parser.add_argument('--host',default='255.255.255.255')
    parser.add_argument('--control-port',type=int,default=45889)
    parser.add_argument('--marker',type=int,default=1)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    if not math.isfinite(args.duration) or args.duration<=0:parser.error('duration must be positive')
    result=benchmark(args.duration,args.host,args.control_port,args.marker)
    rendered=json.dumps(result,indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(rendered)


if __name__=='__main__':main()
