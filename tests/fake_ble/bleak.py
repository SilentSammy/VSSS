"""Radio substitute used only by the subprocess integration tests."""
import asyncio
import json
import os
import time
import struct


def record(event, **details):
    with open(os.environ['BLE_TEST_LOG'], 'a') as stream:
        stream.write(json.dumps(dict(event=event, pid=os.getpid(), time=time.monotonic(), **details)) + '\n')


class Device:
    address = 'test-address'


class BleakScanner:
    @staticmethod
    async def find_device_by_name(name, timeout):
        record('scan', name=name)
        if name == 'missing':
            await asyncio.sleep(60)
        return Device()


drop_once = True


class BleakClient:
    def __init__(self, device, disconnected_callback):
        assert isinstance(device, Device), 'Pass the discovered device, not its address'
        self.callback = disconnected_callback
        self.is_connected = False
        self.services_ready = False
        self.matrix = struct.pack('<9f', 1., 0., 0., 0., 1., 0., 0., 0., 1.)

    async def connect(self):
        self.is_connected = True  # Physical connection can precede GATT readiness.
        record('link')
        await asyncio.sleep(.15)
        self.services_ready = True
        record('ready')

    async def write_gatt_char(self, uuid, data, response):
        global drop_once
        assert self.services_ready and response is True
        await asyncio.sleep(.01)
        if uuid.endswith('def3'):
            self.matrix = bytes(data)
            record('trim_matrix', data=list(data))
            return
        record('write', data=list(data))
        if any(data) and os.environ.get('BLE_TEST_DROP_ON_MOVE') and drop_once:
            drop_once = False
            self.is_connected = False
            record('lost')
            self.callback(self)

    async def read_gatt_char(self, uuid):
        assert self.services_ready
        if uuid.endswith('def3'):
            if os.environ.get('BLE_TEST_LEGACY_TRIM'):
                raise LookupError('Characteristic not found')
            return self.matrix
        raise LookupError('Characteristic not found')

    async def disconnect(self):
        self.is_connected = False
        record('disconnect')
        self.callback(self)
