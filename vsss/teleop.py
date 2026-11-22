"""Guided keyboard and gamepad control of nearby VSSS cars."""
import argparse
from threading import Condition, Thread
import time

from .models import Velocity
from .provider import VSSSProvider

KEYBOARDS = {
    'wasd': ('w', 's', 'd', 'a', 'e', 'q', 'c'),
    'ijkl': ('i', 'k', 'l', 'j', 'o', 'u', 'm'),
}


class LatestCommands:
    """Keep sampling inputs while an acknowledged device write is in flight.

    At most one pending value per car; intermediate joystick positions are
    replaced rather than played back later. Ordinary RPC remains synchronous.
    """
    def __init__(self, cars, channel_factory=None, retry_connections=False):
        self.cars = cars
        self.channel_factory = channel_factory
        self.retry_connections = retry_connections
        self.error = None
        self._condition = Condition()
        self._pending = {}
        self._desired = {}
        self._closed = False
        self._threads = {}

    def submit(self, car_id, command):
        with self._condition:
            if self.error:
                raise RuntimeError('Command writer failed') from self.error
            if self._closed:
                raise RuntimeError('Command writer is closed')
            self._pending[car_id] = command
            self._desired[car_id] = command
            if car_id not in self._threads:
                self._thread = Thread(target=self._run, args=(car_id,), daemon=True)
                self._threads[car_id] = self._thread
                self._thread.start()
            self._condition.notify_all()

    def _run(self, car_id):
        channel, cars = None, self.cars
        try:
            if self.channel_factory:
                channel = self.channel_factory()
                cars = channel.cars
            if self.retry_connections:
                self._run_reconnecting(car_id, cars)
                return
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._closed or car_id in self._pending)
                    if self._closed: return
                    command = self._pending.pop(car_id)
                cars.set_velocity(car_id, command)
        except Exception as error:
            with self._condition:
                self.error = error
                self._pending.pop(car_id, None)
        finally:
            # Each fast car stops without waiting for another stalled radio.
            try: cars.release(car_id)
            except Exception: pass
            if channel is not None: channel.close()

    def _run_reconnecting(self, car_id, cars):
        """Discovery stays off the input thread; reconnect always requires neutral."""
        generation = None
        armed = False
        next_check = 0.
        while True:
            with self._condition:
                if self._closed: return
            try:
                if generation is None:
                    print(f'Looking for car{car_id}...', flush=True)
                    state = cars.connect(car_id)
                    generation = state['generation']
                    with self._condition:
                        if self._closed: return
                        self._pending.pop(car_id, None)
                        armed = self._desired.get(car_id) == Velocity()
                    print(f'car{car_id}: connected. Release controls to neutral before moving.', flush=True)
                    next_check = time.monotonic() + .25
                with self._condition:
                    self._condition.wait_for(lambda: self._closed or car_id in self._pending,
                                             timeout=max(0., next_check-time.monotonic()))
                    if self._closed: return
                    command = self._pending.pop(car_id, None)
                if time.monotonic() >= next_check:
                    state = cars.connection(car_id)
                    if not state['connected'] or state['generation'] != generation:
                        raise ConnectionError(f'car{car_id} disconnected')
                    next_check = time.monotonic() + .25
                if command is not None:
                    if command == Velocity(): armed = True
                    if armed:
                        cars.set_velocity(car_id, command, generation=generation)
            except Exception as error:
                # RPC preserves built-in errors; remote BLE errors include their type.
                if not (isinstance(error, (ConnectionError, OSError)) or
                        type(error).__name__.startswith('Bleak') or str(error).startswith('Bleak')):
                    raise
                generation, armed = None, False
                print(f'car{car_id}: {error}; retrying...', flush=True)
                with self._condition:
                    self._condition.wait_for(lambda: self._closed, timeout=.5)

    def close(self):
        with self._condition:
            self._closed = True
            self._pending.clear()
            self._condition.notify_all()
        # Finish the one in-flight write before the final acknowledged STOP.
        deadline = time.monotonic() + 31
        for thread in self._threads.values():
            thread.join(timeout=max(0., deadline-time.monotonic()))
        if any(thread.is_alive() for thread in self._threads.values()):
            raise TimeoutError('Command writer did not finish')


def assign_inputs(cars, methods, prompt=input):
    """Choose unique cars and input methods before connecting or moving."""
    if len(cars) == 1:
        selected = [method for method in ('wasd', 'gamepad:0') if method in methods]
        if not selected and methods:
            selected = [methods[0]]
        return [(cars[0], tuple(selected))] if selected else []
    remaining = list(cars)
    methods = list(methods)
    assignments = []
    while remaining and methods:
        print('Available cars: ' + ', '.join(map(str, remaining)))
        choice = prompt('Car number (Enter to finish): ').strip()
        if not choice:
            break
        try:
            number = int(choice)
            if number not in remaining:
                raise ValueError()
        except ValueError:
            print('Choose an available car number.')
            continue
        for index, method in enumerate(methods, 1):
            print(f'  {index}) {method}')
        while True:
            choice = prompt(f'Input for car{number} (option number): ').strip()
            if choice.isdigit() and 1 <= int(choice) <= len(methods):
                method = methods.pop(int(choice) - 1)
                break
            print('Choose one of the listed input options.')
        assignments.append((number, (method,)))
        remaining.remove(number)
    return assignments


def velocity(methods, inp):
    commands = []
    for method in methods:
        if method in KEYBOARDS:
            fwd, back, right, left, cw, ccw, boost = KEYBOARDS[method]
            scale = .5 + .5 * inp.get_bipolar_ctrl(high_key=boost)
            command = dict(x=inp.get_bipolar_ctrl(fwd, back),
                           y=-inp.get_bipolar_ctrl(right, left),
                           w=-inp.get_bipolar_ctrl(cw, ccw))
        else:
            index = method.split(':')[1]
            scale = .5 + .5 * inp.get_bipolar_ctrl(high_game=f'RT{index}')
            command = dict(x=inp.get_axis(f'LY{index}'),
                           y=-inp.get_axis(f'LX{index}'),
                           w=-inp.get_axis(f'RX{index}'))
        commands.append({axis: value * scale for axis, value in command.items()})
    return {axis: max(-1., min(1., sum(cmd[axis] for cmd in commands))) for axis in ('x', 'y', 'w')}


def explicit_assignments(specs, methods):
    """Resolve CAR[=INPUT+INPUT] entries and reject ambiguous assignments."""
    assignments = []
    used_cars, used_methods = set(), set()
    for spec in specs:
        number, separator, controls = spec.partition('=')
        if not number.isdigit():
            raise ValueError(f'Invalid car number: {number}')
        number = int(number)
        if number in used_cars:
            raise ValueError(f'car{number} was assigned twice')
        if separator:
            selected = controls.split('+')
        else:
            available = [method for method in methods if method not in used_methods]
            selected = [next((m for m in available if m in KEYBOARDS), ''),
                        next((m for m in available if m.startswith('gamepad:')), '')]
            selected = [m for m in selected if m] or ['']
        for method in selected:
            if method not in methods:
                raise ValueError(f'Input {method!r} is unavailable; available inputs: {", ".join(methods)}')
            if method in used_methods:
                raise ValueError(f'Input {method} was assigned twice')
            used_methods.add(method)
        used_cars.add(number)
        assignments.append((number, tuple(selected)))
    return assignments


def main(argv=None, *, provider_factory=None):
    """Use the same controls with an RPC client or a local provider factory.

    The factory receives the endpoint argument; its result is closed on exit.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', help='Provider discovery JSON path')
    parser.add_argument('--speed', type=float, default=1., help='Multiply normal/boost speeds by this value, (0, 1]')
    parser.add_argument('--refresh', action='store_true', help='ignore the 30-second car discovery cache')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--car', action='append', metavar='ID[=INPUT+INPUT]',
                           help='keep looking for these cars; default: keyboard + gamepad; explicit inputs: wasd, ijkl, gamepad:0, ...')
    selection.add_argument('--auto', action='store_true', help='scan and assign cars in numeric order without prompts')
    selection.add_argument('--list', action='store_true', help='print available car numbers and exit')
    args = parser.parse_args(argv)
    if not 0 < args.speed <= 1:
        parser.error('--speed must be in (0, 1]')
    if args.car and args.refresh:
        parser.error('--refresh applies to discovery, not explicit --car assignments')
    provider = None
    writer = None
    assignments = []
    try:
        provider = (provider_factory or VSSSProvider.connect)(args.endpoint)
        if not provider.describe()['capabilities']['commands']:
            parser.error('playback is read-only')
        numbers = None
        if not args.car:
            if not args.list:
                print('Looking for available cars...', flush=True)
            # Interactive startup always scans; never reuse a cached empty scan.
            numbers = provider.cars.available(refresh=args.refresh or not (args.auto or args.list))
        if args.list:
            print(numbers)
            return
        if numbers == []:
            print('No available cars found. Power on a car and try again; bag mode has no controllable cars.')
            return
        from vsss import input as inp
        import inputs
        methods = list(KEYBOARDS) if inp.KEYBOARD_AVAILABLE else []
        # Reserve the first pad even when absent; input.py handles hot-plugging.
        methods += [f'gamepad:{index}' for index in range(max(1, len(inputs.devices.gamepads)))]
        try:
            if args.car:
                assignments = explicit_assignments(args.car, methods)
            elif args.auto and len(numbers) > 1:
                assignments = explicit_assignments([str(number) for number in numbers[:len(methods)]], methods)
            else:
                assignments = assign_inputs(numbers, methods)
        except ValueError as error:
            parser.error(str(error))
        if not assignments:
            print('No cars assigned.')
            return
        for number, controls in assignments:
            if not args.car:
                provider.cars.connect(number)
            print(f'car{number}: {", ".join(controls)}')
        print('W/S or I/K: forward/back; A/D or J/L: strafe; Q/E or U/O: rotate.')
        print('Gamepad: left stick translates, right stick rotates. C/M or RT: full speed.')
        print('Ctrl+C: stop and exit.')
        # Clear keys used by the assignment prompts before enabling motion.
        inp.pressed_keys.clear()
        sent = {}
        writer = LatestCommands(provider.cars, channel_factory=getattr(provider, 'fork', None),
                                retry_connections=bool(args.car))
        while True:
            started = time.monotonic()
            if writer.error:
                raise RuntimeError(f'Command writer failed: {writer.error}') from writer.error
            for number, controls in assignments:
                command = velocity(controls, inp)
                # Preserve the previous client's 0.05 input resolution.
                rounded = tuple(round(command[axis] * args.speed / .05)
                                for axis in ('x', 'y', 'w'))
                if sent.get(number) != rounded:
                    writer.submit(number, Velocity(*(max(-1., min(1., v * .05)) for v in rounded)))
                    sent[number] = rounded
            time.sleep(max(0., .02 - (time.monotonic() - started)))
    except (KeyboardInterrupt, EOFError):
        print('\nStopping teleoperation.')
    except (ConnectionError, OSError, RuntimeError) as error:
        parser.exit(1, f'{error}\n')
    finally:
        if provider is not None:
            try:
                try:
                    if writer is not None:
                        writer.close()
                finally:
                    for number, _ in assignments:
                        try:
                            provider.cars.release(number)
                        except Exception:
                            pass
            finally:
                provider.close()


if __name__ == '__main__':
    main()
