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
        self.controller = struct.pack('<11f', 4., .55, .4, .35, .2, .5,
                                      .12, .08, .025, 5., 600.)
        self.status = struct.pack('<BBBBfffI', 0, 3, 0, 0, .1, .2, 90., 7)
        self.control = bytes(2)

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
            if not os.environ.get('BLE_TEST_REJECT_CONFIG'):
                self.matrix = bytes(data)
            record('trim_matrix', data=list(data))
            return
        if uuid.endswith('def8'):
            if not os.environ.get('BLE_TEST_REJECT_CONFIG'):
                self.controller = bytes(data)
            record('controller_config', data=list(data))
            return
        if uuid.endswith('def7'):
            self.control = (bytes((0xa5, data[0])) if not os.environ.get('BLE_TEST_OLD_CONTROL')
                            else bytes((0, 0)))
            record('control', data=list(data))
            if data == bytes((3,)) and not os.environ.get('BLE_TEST_OLD_CONTROL'):
                asyncio.get_running_loop().call_later(.15, self._lose_connection)
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
        if uuid.endswith('def8'):
            return self.controller
        if uuid.endswith('def9'):
            return self.status
        if uuid.endswith('def7'):
            return self.control
        raise LookupError('Characteristic not found')

    def _lose_connection(self):
        if self.is_connected:
            self.is_connected = False
            record('lost')
            self.callback(self)

    async def disconnect(self):
        self.is_connected = False
        record('disconnect')
        self.callback(self)
