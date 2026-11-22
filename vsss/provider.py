"""A provider composes interchangeable GPS and command components."""
import uuid

from .components import rpc_method
from .rpc import RPCHost, ProviderClient


class VSSSProvider:
    def __init__(self, gps, cars, *, mode="live"):
        if gps.events is not cars.events:
            raise ValueError("components must share an event hub")
        self.gps, self.cars = gps, cars
        self.events = gps.events
        self.mode = mode
        self.source_id = uuid.uuid4().hex

    @rpc_method
    def describe(self):
        return {"mode": self.mode, "source_id": self.source_id,
                "capabilities": {"positions": True, "commands": self.mode != "playback",
                                 "simulation_edit": self.mode == "simulation"}}

    def serve(self, **settings):
        return RPCHost(self, **settings)

    @staticmethod
    def connect(endpoint=None, *, timeout=30):
        return ProviderClient(endpoint, timeout=timeout)

    def close(self):
        try:
            self.cars.close()
        finally:
            self.gps.close()

    def __enter__(self):
        return self

    def disconnected(self, session):
        """Provider-specific resources owned by a departed client."""
        pass

    def __exit__(self, *args):
        self.close()
