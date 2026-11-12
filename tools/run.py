"""Observe telemetry or record guarded velocity/onboard-waypoint experiments.

Motion is preview-only unless --execute is supplied. Configuration reads are
always performed; --trim and --controller explicitly write and read back JSON.
"""
import argparse
import csv
from contextlib import ExitStack
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

from vsss.car import Car
from vsss.gps import GPS


FIELDS = (
    "elapsed_s", "frame_sequence", "frame_age_ms", "x_m", "y_m", "heading_deg",
    "mode", "fault", "wifi_connected", "pose_fresh", "packet_count",
    "command_x", "command_y", "command_w", "target_x", "target_y", "target_heading",
)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    motion = result.add_mutually_exclusive_group()
    motion.add_argument("--velocity", type=float, nargs=3, metavar=("X", "Y", "W"))
    motion.add_argument("--waypoint", type=float, nargs=3, metavar=("X", "Y", "ANGLE"))
    result.add_argument("--observe", type=float, metavar="SECONDS",
                        help="record without sending a motion command")
    result.add_argument("--duration", type=float, default=30.,
                        help="motion observation limit (default: 30 seconds)")
    result.add_argument("--period", type=float, default=.1)
    result.add_argument("--car", type=int, default=1)
    result.add_argument("--connect-timeout", type=float, default=20.)
    result.add_argument("--max-x", type=float, default=.8)
    result.add_argument("--max-y", type=float, default=1.5)
    result.add_argument("--trim", type=Path,
                        help="JSON matrix (or object containing matrix) to store")
    result.add_argument("--controller", type=Path,
                        help="JSON controller mapping to store; partial updates are allowed")
    result.add_argument("--output", type=Path,
                        help="CSV path; default tools/runs/<timestamp>.csv")
    result.add_argument("--execute", action="store_true",
                        help="enable the requested velocity or waypoint motion")
    return result


def parse_args(argv=None):
    result = parser()
    args = result.parse_args(argv)
    numeric = (args.duration, args.period, args.connect_timeout, args.max_x, args.max_y)
    if (args.car < 0 or not all(math.isfinite(value) and value > 0 for value in numeric)
            or args.observe is not None and (not math.isfinite(args.observe) or args.observe <= 0)):
        result.error("car must be nonnegative and durations/bounds must be positive")
    if args.velocity and any(not math.isfinite(value) or abs(value) > 1 for value in args.velocity):
        result.error("velocity axes must be finite and within [-1, 1]")
    if args.observe is not None and (args.velocity or args.waypoint):
        result.error("observe cannot be combined with a motion command")
    if args.waypoint and (not all(math.isfinite(value) for value in args.waypoint)
                          or abs(args.waypoint[0]) > args.max_x
                          or abs(args.waypoint[1]) > args.max_y):
        result.error("waypoint is nonfinite or outside configured bounds")
    return args


def _read_json(path, field=None):
    value = json.loads(path.read_text(encoding="utf-8"))
    if field and isinstance(value, dict) and field in value:
        value = value[field]
    return value


def _output_path(path):
    if path is not None:
        return path
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return Path(__file__).with_name("runs") / (stamp + ".csv")


def _row(started, gps, pose, status, command, target):
    latest = gps.latest
    values = dict.fromkeys(FIELDS)
    values.update(elapsed_s=time.monotonic() - started,
                  frame_sequence=None if latest is None else latest.frame_sequence,
                  frame_age_ms=None if latest is None else latest.frame_age_ms)
    if pose is not None:
        values.update(x_m=pose.x, y_m=pose.y, heading_deg=pose.angle)
    if status is not None:
        values.update(mode=status.mode_name, fault=status.fault_name,
                      wifi_connected=status.wifi_connected, pose_fresh=status.pose_fresh,
                      packet_count=status.packet_count)
    if command is not None:
        values.update(zip(("command_x", "command_y", "command_w"), command))
    if target is not None:
        values.update(zip(("target_x", "target_y", "target_heading"), target))
    return values


def main(argv=None):
    args = parse_args(argv)
    motion = args.velocity or args.waypoint
    if motion and not args.execute:
        print(json.dumps({"mode": "preview",
                          "velocity": args.velocity,
                          "waypoint": args.waypoint,
                          "duration": args.duration}, indent=2))
        print("No BLE connection or movement. Add --execute to run.")
        return 0

    gps = GPS()
    output = None
    try:
        with Car(args.car, timeout=args.connect_timeout) as car, ExitStack() as cleanup:
            cleanup.callback(car.stop)
            trim = car.read_trim()
            controller = car.read_controller()
            if args.trim:
                trim = car.set_trim(_read_json(args.trim, "matrix"))
            if args.controller:
                controller = car.set_controller(_read_json(args.controller, "controller"))
            status = car.read_status()
            print(json.dumps({"trim_matrix": trim,
                              "controller": controller,
                              "status": asdict(status)}, indent=2))

            seconds = args.observe if args.observe is not None else args.duration if motion else 0
            if seconds <= 0:
                return 0
            initial_pose = car.wait_for_position()
            if abs(initial_pose.x) > args.max_x or abs(initial_pose.y) > args.max_y:
                raise RuntimeError("car starts outside the configured observation bounds")
            path = _output_path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            output = path.open("x", newline="", encoding="utf-8")
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            started = time.monotonic()
            command = tuple(args.velocity) if args.velocity else None
            target = tuple(args.waypoint) if args.waypoint else None
            if target:
                car.go_to(*target)
            next_status = started
            while time.monotonic() - started < seconds:
                pose = car.pose
                if pose is None:
                    raise RuntimeError("pose telemetry became stale")
                if abs(pose.x) > args.max_x or abs(pose.y) > args.max_y:
                    raise RuntimeError("car crossed the configured observation bounds")
                if command:
                    car.set_velocity(*command)
                if time.monotonic() >= next_status:
                    status = car.read_status()
                    next_status = time.monotonic() + .5
                writer.writerow(_row(started, gps, pose, status, command, target))
                output.flush()
                if target and status.mode_name == "fault":
                    raise RuntimeError("onboard waypoint fault: " + status.fault_name)
                if target and status.mode_name == "reached":
                    break
                time.sleep(args.period)
            car.stop()
            print("Recorded:", path.resolve())
            return 0
    finally:
        if output is not None:
            output.close()
        gps.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
