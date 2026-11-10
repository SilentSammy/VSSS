"""Send one waypoint through the public BLE client and observe camera telemetry."""

import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gps import GPS
from mecanum_client import MecanumBLEClient


TARGET = (0.0, 0.5, 90.0)


def angle_error(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


gps = GPS(stale_after=.75)
client = MecanumBLEClient("car1")
try:
    client.connect()
    client.wait_until_connected(15)
    deadline = time.monotonic() + 5
    pose = None
    while pose is None and time.monotonic() < deadline:
        pose = gps.get_car(1)
        time.sleep(.05)
    if pose is None:
        raise RuntimeError(gps.feedback_problem or "marker 1 is unavailable")
    print("start", pose, "target", TARGET, flush=True)
    client.set_waypoint(*TARGET)
    print("BLE waypoint acknowledged", flush=True)

    deadline = time.monotonic() + 25
    next_report = 0
    stable_since = None
    while time.monotonic() < deadline:
        pose = gps.get_car(1)
        now = time.monotonic()
        if pose is None:
            raise RuntimeError(gps.feedback_problem or "marker 1 became unavailable")
        distance = math.hypot(pose.x - TARGET[0], pose.y - TARGET[1])
        heading = angle_error(pose.angle, TARGET[2])
        if now >= next_report:
            print("pose", pose, "distance", round(distance, 3),
                  "heading_error", round(heading, 1), flush=True)
            next_report = now + .25
        if distance <= .04 and heading <= 7:
            stable_since = now if stable_since is None else stable_since
            if now - stable_since >= .7:
                print("reached", pose, flush=True)
                break
        else:
            stable_since = None
        time.sleep(.05)
    else:
        raise RuntimeError("waypoint observation timed out")
finally:
    try:
        if client.is_connected:
            client.stop()
    finally:
        client.disconnect()
        gps.close()
        print("stopped and disconnected", flush=True)
