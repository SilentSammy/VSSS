"""Select ordinary VSSS providers; never duplicate their simulation or IPC."""
from contextlib import ExitStack
import os
from pathlib import Path
from threading import RLock

from vsss import VSSSProvider
from vsss.rpc import default_endpoint
from vsss.scenarios import load_scenario
from vsss.simulation import VSSSSim

ROOT = Path(__file__).resolve().parent.parent


class Environment:
    def __init__(self, endpoint=None, port=38475):
        self.endpoint = Path(endpoint or os.environ.get('VSSS_ENDPOINT') or default_endpoint())
        self.port = port
        self.lock = RLock()
        self.selected = 'car1_ball1'
        self.owned = None
        self.resources = ExitStack()
        self.scenarios = {name: ROOT / 'examples' / f'{name}.jsonl' for name in (
            'car1', 'car1_ball1', 'cars2_ball1', 'cars4_ball1', 'goalie_easy', 'goalie_hard')}

    def options(self):
        labels = ['1 car', '1 car 1 ball', '2 cars 1 ball', '4 cars 1 ball', 'goalie easy', 'goalie hard']
        return [dict(id='live', label='Live'), *[
            dict(id=name, label=label) for name, label in zip(self.scenarios, labels)]]

    def _scenario(self, name):
        return load_scenario(self.scenarios[name])

    def _client(self):
        try:
            return VSSSProvider.connect(self.endpoint, timeout=1)
        except (OSError, ValueError, ConnectionError):
            return None
        except Exception as error:
            # Pyro transport errors mean an endpoint file outlived its provider.
            from Pyro5.errors import CommunicationError
            if isinstance(error, CommunicationError): return None
            raise

    def status(self):
        with self.lock:
            client = self._client()
            if client is None:
                return dict(selected=self.selected, mode=None, source_id=None, options=self.options())
            with client:
                description = client.describe()
            if description['mode'] == 'live': self.selected = 'live'
            return dict(selected=self.selected, mode=description['mode'],
                        source_id=description['source_id'], options=self.options())

    def select(self, name, *, reset=True):
        if name != 'live' and name not in self.scenarios:
            raise ValueError('Unknown environment')
        with self.lock:
            desired = 'live' if name == 'live' else 'simulation'
            client = self._client()
            if client is not None:
                with client:
                    description = client.describe()
                    if description['mode'] == desired:
                        if desired == 'simulation' and reset:
                            client.load_scenario(self._scenario(name))
                        self.selected = name
                        return self.status()
                    if self.owned != description['source_id']:
                        raise ValueError('Stop the provider running in another terminal before changing Live/Simulation mode.')
                self.resources.close()
                self.owned = None
            self.resources.close()
            resources = ExitStack()
            try:
                if desired == 'simulation':
                    provider = resources.enter_context(VSSSSim(self._scenario(name)).start())
                else:
                    from vsss.events import EventHub
                    from vsss.live import GPSLive, CarCmdLive
                    events = EventHub()
                    gps = GPSLive(events)
                    try:
                        provider = VSSSProvider(gps, CarCmdLive(events))
                    except Exception:
                        gps.close()
                        raise
                    resources.enter_context(provider)
                resources.enter_context(provider.serve(endpoint=self.endpoint, port=self.port))
            except Exception:
                resources.close()
                raise
            self.resources = resources
            self.owned, self.selected = provider.source_id, name
            return self.status()

    def ensure(self):
        """Start on first Run, keeping an already selected world's current state."""
        return self.select(self.selected, reset=False)

    def close(self):
        with self.lock:
            self.resources.close()
            self.owned = None
