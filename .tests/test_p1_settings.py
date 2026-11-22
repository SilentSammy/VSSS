import asyncio
from copy import deepcopy
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mp.robot_config import DEFAULTS
from vsss import *
from vsss import _ble_protocol as p
from vsss.live import CarCmdLive
from vsss.recording import Recorder
from vsss.simulation import VSSSSim


class FakeBLE:
    def __init__(self):
        self.writes, self.devices = [], {}
        outer = self
        class Scanner:
            @staticmethod
            async def find_device_by_filter(*args, **kwargs): return SimpleNamespace(name='car1')
            @staticmethod
            async def discover(**kwargs): return {}
        class Client:
            def __init__(self, device, **kwargs):
                self.is_connected = False
                self.config = outer.devices.setdefault(device.name, p.normalize_robot_config(deepcopy(DEFAULTS)))
                self.control, self.chunks = b'\x00\x00', bytearray()
            async def connect(self): self.is_connected = True
            async def disconnect(self): self.is_connected = False
            async def write_gatt_char(self, uuid, value, response):
                value = bytes(value)
                outer.writes.append((uuid, value))
                if uuid == p.TRIM_MATRIX_UUID: self.config['trim_matrix'] = list(p.TRIM_STRUCT.unpack(value))
                elif uuid == p.CONTROLLER_UUID: self.config['controller'] = dict(zip(p.CONTROLLER_KEYS, p.CONTROLLER_STRUCT.unpack(value)))
                elif uuid == p.CONFIG_UUID:
                    offset, total = struct.unpack('<HH', value[1:5])
                    if offset == 0: self.chunks = bytearray(total)
                    self.chunks[offset:offset+len(value)-5] = value[5:]
                    if offset+len(value)-5 == total:
                        updated = p.normalize_robot_config(json.loads(self.chunks))
                        self.config.clear(); self.config.update(updated)
                elif uuid == p.CONTROL_UUID: self.control = bytes([0xa5, value[0]])
            async def read_gatt_char(self, uuid):
                if uuid == p.TRIM_MATRIX_UUID: return p.TRIM_STRUCT.pack(*self.config['trim_matrix'])
                if uuid == p.CONTROLLER_UUID: return p.CONTROLLER_STRUCT.pack(*(self.config['controller'][k] for k in p.CONTROLLER_KEYS))
                if uuid == p.CONFIG_UUID: return json.dumps(self.config).encode()
                if uuid == p.CONTROL_UUID: return self.control
                if uuid == p.STATUS_UUID: return p.STATUS_STRUCT.pack(0, 3, 0, 0, .1, .2, 90, 42)
                return b''
        self.module = SimpleNamespace(BleakClient=Client, BleakScanner=Scanner)


class SettingsTest(unittest.TestCase):
    def test_live_read_write_readback_recovery_and_secret_exclusion(self):
        fake = FakeBLE()
        with patch.dict(sys.modules, {'bleak': fake.module}), tempfile.TemporaryDirectory() as directory:
            hub = EventHub()
            with VSSSProvider(GPSState(hub), CarCmdLive(hub)) as provider:
                endpoint, bag = Path(directory)/'rpc.json', Path(directory)/'recording.jsonl'
                with provider.serve(endpoint=endpoint, port=0), VSSSProvider.connect(endpoint) as client:
                    client.cars.connect(1)
                    with Recorder(client, bag):
                        trim = [[1., .1, 0], [0, 1., 0], [0, 0, 1.]]
                        actual = client.cars.set_trim(1, trim)
                        self.assertAlmostEqual(actual[0][1], .1, places=6)
                        self.assertEqual(client.cars.read_trim(1), actual)
                        controller = client.cars.set_controller(1, dict(max_speed=.3))
                        self.assertAlmostEqual(controller['max_speed'], .3, places=6)
                        self.assertEqual(client.cars.read_controller(1), controller)
                        configuration = client.cars.read_robot_config(1)
                        configuration['wifi_password'] = 'NEVER-RECORD-THIS-SECRET'
                        self.assertEqual(client.cars.set_robot_config(1, configuration), configuration)
                        client.cars.connect(1)
                        self.assertTrue(client.cars.reconnect_telemetry(1))
                        self.assertTrue(client.cars.restart(1))
                        self.assertFalse(client.cars.connection(1)['connected'])
                        client.cars.connect(1)
                        self.assertTrue(client.cars.status(1).pose_fresh)
                    self.assertNotIn('NEVER-RECORD-THIS-SECRET', bag.read_text())
                self.assertTrue(any(uuid == p.CONFIG_UUID for uuid, _ in fake.writes))

    def test_live_waypoint_uses_firmware_degrees_and_lease_not_velocity_heartbeat(self):
        fake = FakeBLE()
        with patch.dict(sys.modules, {'bleak': fake.module}):
            cars = CarCmdLive(EventHub())
            try:
                cars.go_to(1, Pose(.1, .2, 90))
                payload = next(value for uuid, value in fake.writes if uuid == p.WAYPOINT_UUID)
                self.assertEqual(p.WAYPOINT_STRUCT.unpack(payload)[2], 90.)
                # Emulate the firmware's only unit conversion and verify all
                # cardinal headings, including signed angles and a full turn.
                import math
                from mp.onboard_controller import OnboardController, wrap_angle
                for heading in (0.,90.,180.,-90.,270.,360.):
                    cars.go_to(1,Pose(.1,.2,heading))
                    payload = [value for uuid,value in fake.writes if uuid==p.WAYPOINT_UUID][-1]
                    x,y,wire_heading = p.WAYPOINT_STRUCT.unpack(payload)
                    self.assertEqual(wire_heading,heading)
                    controller = OnboardController()
                    controller.set_target(x,y,math.radians(wire_heading))
                    self.assertAlmostEqual(wrap_angle(controller.target[2]-math.radians(heading)),0.)
                    # A car already facing the requested field heading must
                    # receive zero turn, just like the simulator controller.
                    self.assertAlmostEqual(controller.update((x,y,heading),0)[2],0.)
                async def heartbeat():
                    cars._last_write[1] -= 1
                    await cars._heartbeat_car(1)
                cars._call(heartbeat())
                self.assertEqual(fake.writes[-1], (p.LEASE_UUID, b'\x01'))
                cars.set_velocity(1, Velocity(.1))
                self.assertNotIn(1, cars._position_mode)
            finally: cars.close()

    def test_sim_configuration_matches_api_and_does_not_open_start_gate(self):
        with VSSSSim(wait_for_control=True) as sim:
            sim.cars.connect(1)
            sim.cars.status(1)
            sim.cars.set_controller(1, dict(max_speed=.3))
            self.assertTrue(sim._waiting)
            self.assertEqual(sim.cars.read_controller(1)['max_speed'], .3)
            sim.cars.set_trim(1, [[1., 0, 0], [0, .9, 0], [0, 0, 1.]])
            self.assertEqual(sim.cars.read_trim(1)[1][1], .9)
            config = sim.cars.read_robot_config(1)
            self.assertEqual(sim.cars.set_robot_config(1, config), config)
            before = sim.cars.connection(1)['generation']
            sim.cars.restart(1)
            self.assertGreater(sim.cars.connection(1)['generation'], before)
