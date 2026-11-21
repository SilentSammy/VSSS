"""Direct live teleop: same controls and BLE adapter, no RPC server or client.

Stop vsss-live first so it releases the car's BLE connection. Then run:
    python examples/teleop_direct.py --car 1
Compare with vsss-live + vsss-teleop --car 1 using the same controls/speed.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vsss import EventHub, VSSSProvider
from vsss.live import CarCmdLive, GPSLive
from vsss.teleop import main as teleop


def local_provider(endpoint=None):
    if endpoint is not None:
        raise ValueError('--endpoint does not apply to direct teleop')
    events = EventHub()
    gps = GPSLive(events)  # Same UDP/HTTP defaults as vsss-live.
    try:
        cars = CarCmdLive(events)
    except BaseException:
        gps.close()
        raise
    # No .connect() or .serve(): calls use these ordinary local objects.
    return VSSSProvider(gps, cars)


if __name__ == '__main__':
    print('Direct teleop: local live provider, no IPC.', flush=True)
    teleop(provider_factory=local_provider)
