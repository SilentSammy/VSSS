"""Read car configuration/status through the existing BLE worker; never move."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mecanum_client import MecanumBLEClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--car', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=20)
    args = parser.parse_args()
    client = MecanumBLEClient('car%d' % args.car)
    try:
        client.connect()
        client.wait_until_connected(args.timeout)
        print('trim:', json.dumps(client.read_trim()), flush=True)
        print('controller:', json.dumps(client.read_controller(), indent=2), flush=True)
        first = client.read_status()
        print('status 1:', json.dumps(asdict(first)), flush=True)
        time.sleep(2)
        second = client.read_status()
        print('status 2:', json.dumps(asdict(second)), flush=True)
        print('mode=%s fault=%s wifi=%s fresh_pose=%s packet_delta=%d' % (
            second.mode_name, second.fault_name, second.wifi_connected,
            second.pose_fresh, second.packet_count - first.packet_count), flush=True)
    finally:
        client.disconnect()
        print('BLE worker closed.', flush=True)


if __name__ == '__main__':
    main()
