"""Disposable student interpreter. Robot I/O is owned by the live supervisor."""
import builtins
from collections import namedtuple
import json
import math
import sys
import time
from types import SimpleNamespace

wire_in, wire_out = sys.stdin, sys.stdout


def send(message):
    wire_out.write(json.dumps(message) + '\n')
    wire_out.flush()


def call(operation, *args, **kwargs):
    send(dict(kind='call', operation=operation, args=args, kwargs=kwargs))
    line = wire_in.readline()
    if not line:
        raise RuntimeError('Robot supervisor disconnected.')
    value = json.loads(line)['value']
    if isinstance(value, dict) and 'fields' in value:
        record = namedtuple(value['type'], value['fields'])
        return record(**value['fields'])
    if operation in {'position', 'go_to'} and value is not None:
        return tuple(value)
    return value


class Car:
    def __init__(self, number):
        call('connect', number)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        call('disconnect')

    @property
    def pose(self):
        return call('pose')

    @property
    def position(self):
        return call('position')

    @property
    def is_connected(self):
        return call('is_connected')

    def set_velocity(self, x=0., y=0., w=0.):
        return call('set_velocity', x=x, y=y, w=w)

    def go_to(self, x, y, angle):
        return call('go_to', x, y, angle)

    def stop(self):
        return call('stop')

    def read_status(self):
        return call('read_status')

    def wait_for_position(self, timeout=10):
        # Polling stays in the disposable interpreter, so Stop never waits for
        # the full camera timeout before it can regain the robot connection.
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be positive and finite')
        deadline = time.monotonic() + timeout
        while True:
            pose = self.pose
            if pose is not None:
                return pose
            if time.monotonic() >= deadline:
                raise TimeoutError('No fresh camera position; check lab Wi-Fi and marker visibility.')
            time.sleep(.05)


class Output:
    remaining = 16000

    def write(self, text):
        if len(text) > self.remaining:
            raise RuntimeError('Output limit reached (16,000 characters).')
        self.remaining -= len(text)
        send(dict(kind='output', text=text))
        return len(text)

    def flush(self):
        pass


def student_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name == 'car' and tuple(fromlist) == ('Car',):
        return SimpleNamespace(Car=Car)
    if level == 0 and name == 'time' and tuple(fromlist) == ('sleep',):
        return SimpleNamespace(sleep=time.sleep)
    raise ImportError('Only Car and sleep are available in the block runner.')


code = json.loads(wire_in.readline())['code']
sys.stdout = Output()
allowed = {name: getattr(builtins, name) for name in
           ('print', 'range', 'str', 'int', 'float', 'len', 'abs', 'round', 'min', 'max', 'sum')}
allowed['__import__'] = student_import
error = ''
try:
    exec(compile(code, '<blocks>', 'exec'), {'__builtins__': allowed})
except Exception as exc:
    error = f'{type(exc).__name__}: {exc}'
send(dict(kind='done', error=error))
