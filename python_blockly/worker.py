"""Disposable Python interpreter; all devices are owned by the supervisor."""
import builtins
from collections import namedtuple
from functools import wraps
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api_registry import CAR_METHODS, CAR_PROPERTIES, GPS_METHODS, GPS_PROPERTIES, INPUT_METHODS, INPUT_PROPERTIES

wire_in, wire_out = sys.stdin, sys.stdout


def send(message):
    wire_out.write(json.dumps(message) + '\n')
    wire_out.flush()


def decode(value):
    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    if '$record' in value:
        fields = {k: decode(v) for k, v in value['fields'].items()}
        return namedtuple(value['$record'], fields)(**fields)
    if '$map' in value:
        return {decode(k): decode(v) for k, v in value['$map']}
    if '$tuple' in value:
        return tuple(decode(v) for v in value['$tuple'])
    raise ValueError('Unsupported supervisor response.')


def call(operation, *args, target=None, member=None, **kwargs):
    send(dict(kind='call', operation=operation, target=target, member=member, args=args, kwargs=kwargs))
    line = wire_in.readline()
    if not line:
        raise RuntimeError('Robot supervisor disconnected.')
    return decode(json.loads(line)['value'])


class Field:
    def __getattr__(self, name):
        if name in GPS_PROPERTIES:
            return call('gps_get', member=name)
        if name in GPS_METHODS:
            return lambda *args, **kwargs: call('gps_call', *args, member=name, **kwargs)
        raise AttributeError(name)


gps = Field()



class Input:
    def __getattr__(self, name):
        if name in INPUT_METHODS:
            return lambda *args, **kwargs: call('input_call', *args, member=name, **kwargs)
        if name in INPUT_PROPERTIES:
            return call('input_get', member=name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        raise AttributeError('Student input settings are read-only.')


student_input = Input()


class Car:
    def __init__(self, number, *, timeout=15):
        self._handle = call('connect', number, timeout=timeout)

    def __getattr__(self, name):
        if name in CAR_PROPERTIES:
            return call('car_get', target=self._handle, member=name)
        if name in CAR_METHODS:
            return lambda *args, **kwargs: call('car_call', *args, target=self._handle, member=name, **kwargs)
        raise AttributeError(name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        call('car_call', target=self._handle, member='close')

    def wait_for_position(self, timeout=10):
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


def run(function):
    # The supervisor owns resources across the process boundary and implements
    # the real @run's cleanup even if this interpreter is killed, not just on a
    # Python exception. Exported code uses the actual vsss.run decorator.
    return function


class Output:
    def write(self, text):
        send(dict(kind='output', text=text[-16000:]))
        return len(text)

    def flush(self):
        pass


def student_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name == 'vsss' and set(fromlist) <= {'Car', 'gps', 'run', 'input'}:
        return SimpleNamespace(Car=Car, gps=gps, run=run, input=student_input)
    if level == 0 and name == 'car' and tuple(fromlist) == ('Car',):
        return SimpleNamespace(Car=Car)
    if level == 0 and name == 'time' and set(fromlist) <= {'sleep', 'monotonic'}:
        return SimpleNamespace(sleep=time.sleep, monotonic=time.monotonic)
    raise ImportError('Only the approved VSSS API, sleep and monotonic are available.')


code = json.loads(wire_in.readline())['code']
sys.stdout = Output()
allowed = {name: getattr(builtins, name) for name in
           ('print', 'range', 'str', 'int', 'float', 'bool', 'len', 'abs', 'round', 'min', 'max',
            'sum', 'dict', 'list', 'tuple', 'enumerate', 'sorted', 'any', 'all',
            'RuntimeError', 'TimeoutError', 'ValueError')}
allowed['__import__'] = student_import
error = ''
try:
    exec(compile(code, '<blocks>', 'exec'), {'__builtins__': allowed, '__name__': '__main__'})
except Exception as exc:
    error = f'{type(exc).__name__}: {exc}'
send(dict(kind='done', error=error))
