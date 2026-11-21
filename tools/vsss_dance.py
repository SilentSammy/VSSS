"""Check a fleet by alternating rotation; discover newly powered cars continuously."""
import argparse
import math
from queue import Empty, Queue
from threading import Event, Thread
import time

from vsss._ble_client import MecanumBLEClient
from vsss.provider import available_cars


def discover(events, done, interval):
    while not done.is_set():
        try:
            events.put(('cars', available_cars(refresh=True)))
        except Exception as error:
            events.put(('error', str(error)))
        done.wait(interval)


def dance(args):
    events, done = Queue(), Event()
    clients, connected, sent = {}, {}, {}
    scanner = None
    started = time.monotonic()
    if args.car:
        events.put(('cars', sorted(set(args.car))))
    else:
        scanner = Thread(target=discover, args=(events, done, args.scan_interval), daemon=True)
        scanner.start()
    print('Alternating rotation; Ctrl+C stops all cars. This checks BLE and visible motor motion, not GPS.', flush=True)
    try:
        while not args.duration or time.monotonic() - started < args.duration:
            try:
                while True:
                    kind, value = events.get_nowait()
                    if kind == 'error':
                        print(f'Discovery failed; retrying: {value}', flush=True)
                        continue
                    for number in value:
                        if number not in clients:
                            client = MecanumBLEClient(device_name=f'car{number}')
                            clients[number] = client
                            print(f'car{number}: discovered; connecting', flush=True)
                            try:
                                client.connect()
                            except Exception as error:
                                print(f'car{number}: connection failed: {error}', flush=True)
                                client.disconnect()
                                del clients[number]
            except Empty:
                pass
            phase = int((time.monotonic() - started) / args.interval) % 2
            for number, client in list(clients.items()):
                ready = client.is_connected
                if connected.get(number, False) != ready:
                    print(f'car{number}: ' + ('connected; dancing' if ready else 'disconnected; waiting for reconnect'), flush=True)
                    connected[number] = ready
                if not ready:
                    sent.pop(number, None)
                    continue
                state = (client.connection_generation, phase)
                if sent.get(number) != state:
                    try:
                        client.set_velocity(dict(x=0., y=0., w=args.speed * (1 if phase == 0 else -1)))
                        sent[number] = state
                    except (ConnectionError, OSError, RuntimeError) as error:
                        print(f'car{number}: command failed: {error}', flush=True)
            time.sleep(.02)
    except KeyboardInterrupt:
        print('\nStopping fleet.', flush=True)
    finally:
        done.set()
        # Issue stops to every car before waiting for acknowledgements/cleanup.
        for number, client in clients.items():
            try:
                if client.is_connected:
                    client.set_velocity(dict(x=0., y=0., w=0.))
            except Exception as error:
                print(f'car{number}: zero command failed: {error}', flush=True)
        for number, client in clients.items():
            try:
                client.stop()
            except Exception as error:
                print(f'car{number}: stop not acknowledged: {error}', flush=True)
            finally:
                try:
                    client.disconnect()
                except Exception as error:
                    print(f'car{number}: disconnect failed: {error}', flush=True)
        if scanner is not None:
            scanner.join(timeout=.2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--car', type=int, action='append', help='only these IDs; repeat for multiple cars; disables discovery')
    parser.add_argument('--speed', type=float, default=.2, help='rotation magnitude in (0, 1], default .2')
    parser.add_argument('--interval', type=float, default=1., help='seconds before reversing direction, default 1')
    parser.add_argument('--scan-interval', type=float, default=5., help='seconds between completed scans, default 5')
    parser.add_argument('--duration', type=float, default=0., help='stop after this many seconds; 0 runs until Ctrl+C')
    args = parser.parse_args(argv)
    for name in ('speed', 'interval', 'scan_interval', 'duration'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name != 'duration' and value == 0):
            parser.error(f'--{name.replace("_", "-")} must be finite and ' + ('nonnegative' if name == 'duration' else 'positive'))
    if args.speed > 1:
        parser.error('--speed must not exceed 1')
    if args.car and any(number < 0 for number in args.car):
        parser.error('--car must be nonnegative')
    dance(args)


if __name__ == '__main__':
    main()
