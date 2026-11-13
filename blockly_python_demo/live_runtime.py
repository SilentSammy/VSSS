"""Own all real Car connections outside disposable student code."""
import json
from pathlib import Path
import subprocess
import sys
from threading import Event, Lock, RLock, Thread
import time
import uuid

from api_registry import CAR_METHODS, CAR_PROPERTIES, GPS_METHODS, GPS_PROPERTIES, INPUT_METHODS, INPUT_PROPERTIES
from wire import encode

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
ACTIVE = {'running', 'stopping'}


def real_car(number, **kwargs):
    from vsss import Car
    return Car(number, **kwargs)


def real_gps():
    from vsss import gps
    return gps


def real_input():
    from vsss import input
    return input


def real_cleanup():
    from vsss import Car, gps
    try:
        Car.close_all()
    finally:
        gps.close()  # The lazy facade does not create a receiver here.


class CarEntry:
    def __init__(self, car):
        self.car = car
        self.generation = car.connection_generation
        self.action = RLock()
        self.closing = False
        self.closed = False
        self.cleanup_thread = None


class LiveRun:
    def __init__(self, code, car_factory, gps_factory, input_factory, cleanup, lease_seconds, max_seconds):
        self.id = uuid.uuid4().hex
        self.code = code
        self.car_factory, self.gps_factory, self.input_factory = car_factory, gps_factory, input_factory
        self.resource_cleanup = cleanup
        self.lease_seconds, self.max_seconds = lease_seconds, max_seconds
        self.started = self.heartbeat = time.monotonic()
        self.lock = RLock()
        self.cancel = Event()
        self.state = 'running'
        self.output = self.error = self.reason = ''
        self.process = None
        self.cars = {}
        self.gps = self.input = None
        self.thread = Thread(target=self.execute, daemon=True, name='Blockly-live-run')

    def snapshot(self, renew=False):
        with self.lock:
            if renew and not self.cancel.is_set():
                self.heartbeat = time.monotonic()
            return dict(run_id=self.id, state=self.state, output=self.output,
                        error=self.error, reason=self.reason, live=True)

    def append(self, text):
        with self.lock:
            self.output = (self.output + text)[-16000:]

    def add_error(self, message):
        with self.lock:
            self.error += ('\n' if self.error else '') + message

    def stop(self, reason):
        with self.lock:
            if self.state not in ACTIVE:
                return
            if not self.cancel.is_set():
                self.reason = reason
            self.cancel.set()
            self.state = 'stopping'
            process = self.process
            entries = list(self.cars.values())
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass
        # Each connected car gets its own cleanup thread. A blocked operation or
        # another car still connecting must not delay stopping the other cars.
        for entry in entries:
            self.begin_close(entry)

    def close_entry(self, entry):
        with entry.action:
            if entry.closed:
                return
            entry.closing = True
            try:
                entry.car.stop()
            except Exception as exc:
                self.add_error(f'car{entry.car.number}: stop was not acknowledged: {exc}')
            try:
                entry.car.close()
                entry.closed = True
            except Exception as exc:
                self.add_error(f'car{entry.car.number}: disconnect failed: {exc}')

    def begin_close(self, entry):
        with self.lock:
            if entry.cleanup_thread is None:
                entry.closing = True
                entry.cleanup_thread = Thread(target=self.close_entry, args=(entry,), daemon=True)
                entry.cleanup_thread.start()
            return entry.cleanup_thread

    def close_all(self):
        with self.lock:
            entries = list(self.cars.values())
        threads = [self.begin_close(entry) for entry in entries]
        for thread in threads:
            thread.join()

    def field(self):
        if self.gps is None:
            self.gps = self.gps_factory()
        return self.gps

    def dispatch(self, message):
        if self.cancel.is_set():
            raise RuntimeError('Run has been stopped.')
        operation = message['operation']
        args, kwargs = message.get('args', []), message.get('kwargs', {})
        member = message.get('member')
        if operation == 'connect':
            number = args[0]
            if type(number) is not int or number < 0:
                raise ValueError('car number must be a nonnegative integer')
            with self.lock:
                if any(e.car.number == number and not e.closed for e in self.cars.values()):
                    raise ValueError(f'car{number} already has a connection in this program; reuse its variable.')
            self.append(f'Connecting to car{number}...\n')
            car = self.car_factory(*args, **kwargs)
            token, entry = uuid.uuid4().hex, CarEntry(car)
            with self.lock:
                self.cars[token] = entry
            if self.cancel.is_set():
                self.begin_close(entry)
                raise RuntimeError('Run was stopped while connecting.')
            self.append(f'car{number} connected.\n')
            return token
        if operation in {'gps_get', 'gps_call'}:
            gps = self.field()
            if operation == 'gps_get' and member in GPS_PROPERTIES:
                return getattr(gps, member)
            if operation == 'gps_call' and member in GPS_METHODS:
                return getattr(gps, member)(*args, **kwargs)
            raise ValueError('Unsupported field operation.')
        if operation in {'input_call', 'input_get'}:
            if self.input is None:
                self.input = self.input_factory()
            if operation == 'input_call' and member in INPUT_METHODS:
                if member == 'get_axis' and (kwargs.get('normalize', True) is not True or len(args) > 1 and args[1] is not True):
                    raise ValueError('Student input blocks use normalized axes only.')
                return getattr(self.input, member)(*args, **kwargs)
            if operation == 'input_get' and member in INPUT_PROPERTIES:
                return getattr(self.input, member)
            raise ValueError('Unsupported input operation.')
        if operation not in {'car_get', 'car_call'}:
            raise ValueError('Unsupported robot operation.')
        with self.lock:
            entry = self.cars.get(message.get('target'))
        if entry is None:
            raise ConnectionError('Unknown car connection.')
        with entry.action:
            if self.cancel.is_set() or entry.closing or entry.closed:
                raise ConnectionError('This car connection has ended.')
            car = entry.car
            if operation == 'car_get' and member in CAR_PROPERTIES:
                return getattr(car, member)
            if operation == 'car_call' and member == 'close':
                self.close_entry(entry)  # Internal context-manager cleanup only.
                return None
            if operation != 'car_call' or member not in CAR_METHODS:
                raise ValueError('Unsupported car method.')
            if not car.is_connected or car.connection_generation != entry.generation:
                raise ConnectionError(f'car{car.number} connection lost or replaced. Start a new run.')
            return getattr(car, member)(*args, **kwargs)

    def watch(self):
        while not self.cancel.wait(.1):
            with self.lock:
                if self.state not in ACTIVE:
                    return
                heartbeat = self.heartbeat
                entries = list(self.cars.values())
            now = time.monotonic()
            if now - heartbeat > self.lease_seconds:
                self.stop('Editor disconnected or stopped responding.')
            elif self.max_seconds is not None and now - self.started > self.max_seconds:
                self.stop(f'Run time limit reached ({self.max_seconds:g} seconds).')
            else:
                for entry in entries:
                    if entry.closing or entry.closed:
                        continue
                    try:
                        car = entry.car
                        if not car.is_connected or car.connection_generation != entry.generation:
                            if not entry.closing:
                                self.stop(f'car{car.number} connection lost or replaced. All cars are stopping.')
                                break
                    except Exception as exc:
                        self.stop(f'Robot monitoring failed: {exc}')
                        break

    def execute(self):
        done = False
        try:
            process = subprocess.Popen([sys.executable, '-I', '-u', str(ROOT / 'worker.py')],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding='utf-8', errors='replace', cwd=ROOT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            with self.lock:
                self.process = process
            Thread(target=self.watch, daemon=True, name='Blockly-watchdog').start()
            if self.cancel.is_set():
                return
            process.stdin.write(json.dumps({'code': self.code}) + '\n')
            process.stdin.flush()
            for line in process.stdout:
                if self.cancel.is_set():
                    break
                message = json.loads(line)
                if message['kind'] == 'output':
                    self.append(message['text'])
                elif message['kind'] == 'done':
                    done = True
                    if message.get('error'):
                        self.add_error(message['error'])
                    break
                elif message['kind'] == 'call':
                    try:
                        value = encode(self.dispatch(message))
                    except Exception as exc:
                        if not self.cancel.is_set():
                            self.add_error(f'{type(exc).__name__}: {exc}')
                        break
                    if self.cancel.is_set():
                        break
                    process.stdin.write(json.dumps({'value': value}) + '\n')
                    process.stdin.flush()
                else:
                    raise ValueError('Invalid worker message.')
            if not done and not self.cancel.is_set() and not self.error:
                raise RuntimeError('Student worker exited unexpectedly.')
        except Exception as exc:
            if not self.cancel.is_set():
                self.add_error(f'{type(exc).__name__}: {exc}')
        finally:
            self.cancel.set()
            process = self.process
            try:
                if process is not None:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=5)
                    for pipe in (process.stdin, process.stdout, process.stderr):
                        try:
                            pipe.close()
                        except OSError:
                            pass
            except Exception as exc:
                self.add_error(f'Student interpreter cleanup failed: {exc}')
            finally:
                self.close_all()
                try:
                    self.resource_cleanup()
                except Exception as exc:
                    self.add_error(f'Resource cleanup failed: {exc}')
                with self.lock:
                    self.state = 'failed' if self.error else ('stopped' if self.reason else 'finished')


class LiveRunner:
    def __init__(self, car_factory=real_car, *, gps_factory=real_gps, input_factory=real_input,
                 cleanup=real_cleanup, lease_seconds=5, max_seconds=None):
        self.factories = car_factory, gps_factory, input_factory, cleanup
        self.lease_seconds, self.max_seconds = lease_seconds, max_seconds
        self.lock = Lock()
        self.current = None

    def start(self, code):
        with self.lock:
            if self.current and self.current.snapshot()['state'] in ACTIVE:
                raise ValueError('A program is already running or stopping.')
            self.current = LiveRun(code, *self.factories, self.lease_seconds, self.max_seconds)
            self.current.thread.start()
            return self.current.snapshot()

    def status(self, run_id=None):
        with self.lock:
            run = self.current
            if run_id and (run is None or run.id != run_id):
                raise ValueError('This run is no longer current.')
            return run.snapshot(renew=bool(run_id)) if run else dict(state='idle', output='', error='', live=True)

    def stop(self, run_id=None, reason='Stopped by user.'):
        with self.lock:
            run = self.current
            if run and (not run_id or run.id == run_id):
                run.stop(reason)
            return run.snapshot() if run else dict(state='idle', output='', error='', live=True)

    def shutdown(self):
        self.stop(reason='Server shutting down.')
        if self.current:
            self.current.thread.join(timeout=75)
