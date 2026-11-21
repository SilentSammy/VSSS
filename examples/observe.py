"""One client for live, playback or simulation. Run from the checkout root."""
import argparse
from pathlib import Path
import sys
from threading import Event

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vsss import VSSSProvider

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--endpoint')
args = parser.parse_args()

with VSSSProvider.connect(args.endpoint) as provider:
    print(provider.describe())
    print(provider.gps.state())
    provider.gps.on_change = lambda state: print('positions:', state)
    provider.cars.on_change = lambda command: print('command:', command)
    try:
        Event().wait()
    except KeyboardInterrupt:
        pass
