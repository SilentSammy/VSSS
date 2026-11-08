"""BLE owner process. stdin carries commands; stdout carries status messages.

The student script owns stdin's other end. EOF means that script has exited,
including os._exit() or an IDE stop, so we stop and disconnect immediately.
"""
import asyncio
import json
import sys
from threading import Thread
from bleak import BleakClient, BleakScanner

VELOCITY_UUID = '12345678-1234-5678-1234-56789abcdef1'
HEARTBEAT_SECONDS = 0.2
WRITE_TIMEOUT = 1.0


class Commands:
    """Only keep the newest velocity, rather than queueing old movements."""
    def __init__(self):
        self.closed = asyncio.Event()
        self.changed = asyncio.Event()
        self.ready = False
        self.velocity = bytes(3)
        self.sequence = 0
        self.force = False

    def update(self, message):
        if self.ready:
            self.velocity = bytes(message['velocity'])
            self.sequence = message['sequence']
            self.force = message['force']
            self.changed.set()

    def report(self, **message):
        try:
            print(json.dumps(message), flush=True)
        except (BrokenPipeError, OSError):
            self.closed.set()


def read_commands(loop, commands):
    try:
        for line in sys.stdin:
            loop.call_soon_threadsafe(commands.update, json.loads(line))
    finally:
        # This thread is only a pipe reader; all BLE work stays on asyncio.
        try:
            loop.call_soon_threadsafe(commands.closed.set)
        except RuntimeError:
            pass  # The event loop has already finished shutting down.


async def write(client, velocity):
    await asyncio.wait_for(
        client.write_gatt_char(VELOCITY_UUID, velocity, response=True), WRITE_TIMEOUT)


async def control(client, commands):
    # Never resume an old movement after reconnecting.
    commands.velocity = bytes(3)
    commands.sequence = 0
    commands.force = False
    await write(client, bytes(3))
    commands.ready = True
    commands.report(ready=True, message='Connected')
    last_velocity = bytes(3)
    last_sequence = 0
    last_write = asyncio.get_running_loop().time()
    while client.is_connected:
        commands.changed.clear()
        velocity, sequence = commands.velocity, commands.sequence
        elapsed = asyncio.get_running_loop().time() - last_write
        if (velocity != last_velocity or elapsed >= HEARTBEAT_SECONDS
                or (commands.force and sequence != last_sequence)):
            await write(client, velocity)
            last_velocity, last_sequence = velocity, sequence
            last_write = asyncio.get_running_loop().time()
            commands.report(written=sequence)
        try:
            await asyncio.wait_for(commands.changed.wait(), max(
                .001, HEARTBEAT_SECONDS - (asyncio.get_running_loop().time() - last_write)))
        except asyncio.TimeoutError:
            pass


async def connect_and_control(name, commands):
    while not commands.closed.is_set():
        client = None
        try:
            commands.report(message='Scanning for ' + name)
            device = await BleakScanner.find_device_by_name(name, timeout=10)
            if device is None:
                raise RuntimeError('Robot not found')
            client = BleakClient(device, disconnected_callback=lambda _: commands.changed.set())
            await asyncio.wait_for(client.connect(), timeout=10)
            await control(client, commands)
        except Exception as error:
            commands.report(error=str(error))
        finally:
            commands.ready = False
            commands.velocity = bytes(3)
            commands.report(ready=False)
            if client is not None:
                try:
                    if client.is_connected:
                        await write(client, bytes(3))
                except Exception:
                    pass  # The robot's timeout also stops it if the link is gone.
                try:
                    await asyncio.wait_for(client.disconnect(), timeout=2)
                except Exception:
                    pass
        await asyncio.sleep(.5)


async def main(name):
    commands = Commands()
    Thread(target=read_commands, args=(asyncio.get_running_loop(), commands), daemon=True).start()
    owner = asyncio.create_task(connect_and_control(name, commands))
    try:
        await commands.closed.wait()
    finally:
        owner.cancel()  # Also cancels discovery or connection setup in progress.
        try:
            await owner
        except asyncio.CancelledError:
            pass


if __name__ == '__main__':
    try:
        asyncio.run(main(sys.argv[1]))
    except KeyboardInterrupt:
        pass
