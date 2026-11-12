"""Own the real Car outside student code so cancellation retains BLE control."""
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import subprocess
import sys
from threading import Event, Lock, Thread
import time
import uuid

ROOT = Path(__file__).resolve().parent
ACTIVE = {'running', 'stopping'}


def real_car(number):
    # Use the interpreter/environment that launched the server, including bleak.
    sys.path.insert(0, str(ROOT.parent))
    from car import Car
    from gps import GPS
    receiver = GPS()
    try:
        return Car(number)
    except BaseException:
        receiver.close()
        raise


class LiveRun:
    def __init__(self, code, car_factory, lease_seconds, max_seconds):
        self.id = uuid.uuid4().hex
        self.code = code
        self.car_factory = car_factory
        self.lease_seconds, self.max_seconds = lease_seconds, max_seconds
        self.started = self.heartbeat = time.monotonic()
        self.lock = Lock()
        self.cancel = Event()
        self.state = 'running'
        self.output = self.error = self.reason = ''
        self.process = None
        self.car = None
        self.used_car = False
        self.motion = False
        self.generation = None
        self.thread = Thread(target=self.execute, daemon=True, name='Blockly-live-run')

    def snapshot(self, renew=False):
        with self.lock:
            if renew and not self.cancel.is_set():
                self.heartbeat = time.monotonic()
            return dict(run_id=self.id, state=self.state, output=self.output,
                        error=self.error, reason=self.reason, live=True)

    def append(self, text):
        with self.lock:
            if len(self.output) + len(text) > 16000:
                raise RuntimeError('Output limit reached (16,000 characters).')
            self.output += text

    def stop(self, reason, expected_car=None):
        with self.lock:
            if expected_car is not None and self.car is not expected_car:
                return
            if self.state not in ACTIVE:
                return
            if not self.cancel.is_set():
                self.reason = reason
            self.cancel.set()
            self.state = 'stopping'
            process = self.process
        # Only the student subprocess is killed. The Car and BLE worker remain
        # owned by this process until execute() completes stop/close below.
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass

    def close_car(self):
        with self.lock:
            car = self.car
            self.car = None
            self.motion = False
        if car is None:
            return
        try:
            car.stop()
        except Exception as exc:
            with self.lock:
                self.error += f'\nStop was not acknowledged: {exc}'
        finally:
            try:
                car.close()
            except Exception as exc:
                with self.lock:
                    self.error += f'\nDisconnect failed: {exc}'
            finally:
                try:
                    car.gps.close()
                except Exception as exc:
                    with self.lock:
                        self.error += f'\nCamera receiver cleanup failed: {exc}'

    def dispatch(self, operation, args, kwargs):
        if self.cancel.is_set():
            raise RuntimeError('Run has been stopped.')
        if operation == 'connect':
            if self.used_car:
                raise ValueError('Use one car session per run.')
            self.used_car = True
            self.append(f'Connecting to car{args[0]}...\n')
            car = self.car_factory(*args, **kwargs)
            with self.lock:
                self.car = car
                self.generation = car.connection_generation
            if self.cancel.is_set():
                raise RuntimeError('Run was stopped while connecting.')
            self.append('Connected.\n')
            return None
        if self.car is None:
            raise ConnectionError('No car session is connected.')
        car = self.car
        if operation == 'disconnect':
            self.close_car()
            return None
        if operation in {'pose', 'position', 'is_connected'}:
            result = getattr(car, operation)
        elif operation in {'set_velocity', 'go_to', 'stop', 'read_status'}:
            if not car.is_connected or car.connection_generation != self.generation:
                raise ConnectionError('Robot connection was lost or replaced. Start a new run.')
            if operation in {'set_velocity', 'go_to'}:
                # Validate velocity before classifying it as a moving command.
                moving = operation == 'go_to' or any(args) or any(kwargs.values())
                if moving and car.pose is None:
                    raise RuntimeError('Fresh camera position is required before motion.')
                with self.lock:
                    self.motion = moving
            result = getattr(car, operation)(*args, **kwargs)
            if operation == 'stop':
                with self.lock:
                    self.motion = False
        else:
            raise ValueError('Unsupported robot operation.')
        if is_dataclass(result):
            return {'type': type(result).__name__, 'fields': asdict(result)}
        return result

    def watch(self):
        while not self.cancel.wait(.1):
            with self.lock:
                if self.state not in ACTIVE:
                    return
                heartbeat, car, motion = self.heartbeat, self.car, self.motion
            now = time.monotonic()
            if now - heartbeat > self.lease_seconds:
                self.stop('Editor disconnected or stopped responding.')
            elif now - self.started > self.max_seconds:
                self.stop(f'Run time limit reached ({self.max_seconds:g} seconds).')
            elif car is not None:
                try:
                    if not car.is_connected or car.connection_generation != self.generation:
                        self.stop('Robot connection lost or replaced. Motion will not resume.', expected_car=car)
                    elif motion and car.pose is None:
                        self.stop('Camera position is missing or stale.', expected_car=car)
                except Exception as exc:
                    self.stop(f'Robot monitoring failed: {exc}', expected_car=car)

    def execute(self):
        done = False
        watchdog = Thread(target=self.watch, daemon=True, name='Blockly-watchdog')
        try:
            process = subprocess.Popen(
                [sys.executable, '-I', '-u', str(ROOT / 'worker.py')],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding='utf-8', errors='replace', cwd=ROOT)
            with self.lock:
                self.process = process
            watchdog.start()
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
                    with self.lock:
                        self.error += message.get('error', '')
                    break
                elif message['kind'] == 'call':
                    try:
                        value = self.dispatch(message['operation'], message.get('args', []), message.get('kwargs', {}))
                        response = {'value': value}
                    except Exception as exc:
                        # A robot/API error ends the run; no later commands run.
                        if not self.cancel.is_set():
                            with self.lock:
                                self.error += f'{type(exc).__name__}: {exc}'
                        break
                    if self.cancel.is_set():
                        break
                    process.stdin.write(json.dumps(response) + '\n')
                    process.stdin.flush()
                else:
                    raise ValueError('Invalid worker message.')
            if not done and not self.cancel.is_set() and not self.error:
                raise RuntimeError('Student worker exited unexpectedly.')
        except Exception as exc:
            if not self.cancel.is_set():
                with self.lock:
                    self.error += f'{type(exc).__name__}: {exc}'
        finally:
            # No student command can execute from here onward.
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
                with self.lock:
                    self.error += f'\nStudent interpreter cleanup failed: {exc}'
            finally:
                self.close_car()
                with self.lock:
                    self.state = 'failed' if self.error else ('stopped' if self.reason else 'finished')


class LiveRunner:
    def __init__(self, car_factory=real_car, *, lease_seconds=5, max_seconds=120):
        self.car_factory = car_factory
        self.lease_seconds, self.max_seconds = lease_seconds, max_seconds
        self.lock = Lock()
        self.current = None

    def start(self, code):
        with self.lock:
            if self.current and self.current.snapshot()['state'] in ACTIVE:
                raise ValueError('A program is already running or stopping.')
            self.current = LiveRun(code, self.car_factory, self.lease_seconds, self.max_seconds)
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
            # Stale pagehide requests must never cancel a newer program.
            if run and (not run_id or run.id == run_id):
                run.stop(reason)
            return run.snapshot() if run else dict(state='idle', output='', error='', live=True)

    def shutdown(self):
        self.stop(reason='Server shutting down.')
        if self.current:
            self.current.thread.join(timeout=25)
