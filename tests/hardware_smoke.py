"""Explicit live check: scan, read status and send STOP only. No motion commands."""
import argparse
import json
import tempfile
from pathlib import Path
import sys

# Also usable as python tests/hardware_smoke.py from this checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vsss import EventHub, GPSState, VSSSProvider
from vsss.live import CarCmdLive
from vsss.models import encode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--car', type=int, default=1)
    args = parser.parse_args()
    events = EventHub()
    with tempfile.TemporaryDirectory() as directory:
        endpoint = Path(directory) / 'provider.json'
        with VSSSProvider(GPSState(events), CarCmdLive(events)) as live:
            with live.serve(endpoint=endpoint, port=0):
                with VSSSProvider.connect(endpoint) as client:
                    cars = client.cars.available()
                    print('advertised cars:', cars, flush=True)
                    if args.car not in cars:
                        raise SystemExit(f'car{args.car} was not discovered')
                    print('status:', json.dumps(encode(client.cars.status(args.car))), flush=True)
                    print('stop acknowledged:', client.cars.stop(args.car), flush=True)
