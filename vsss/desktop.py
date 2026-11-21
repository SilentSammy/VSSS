"""Small Tk clients for field display and held-key car control."""
import argparse
import math
from threading import Lock

from .models import Velocity
from .provider import VSSSProvider


def _parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--endpoint', help='Provider discovery JSON path')
    return parser


def draw_world(canvas, state):
    """Draw a centered field with +x right and +y up; return object positions."""
    canvas.delete('all')
    width, height = max(canvas.winfo_width(), 200), max(canvas.winfo_height(), 200)
    scale = min((width - 80) / state.field.width, (height - 100) / state.field.height)
    def point(x, y):
        return width / 2 + x * scale, height / 2 - y * scale
    left, top = point(-state.field.width / 2, state.field.height / 2)
    right, bottom = point(state.field.width / 2, -state.field.height / 2)
    canvas.create_rectangle(left, top, right, bottom, outline='white', width=2)
    canvas.create_line(left, height / 2, right, height / 2, fill='white')
    positions = []
    for objects, radius, color, name in ((state.cars, .07, '#4fc3ff', 'car'),
                                        (state.balls, .02, '#ffad42', 'ball')):
        for obj in objects:
            x, y = point(obj.pose.x, obj.pose.y)
            r = max(5, radius * scale)
            canvas.create_oval(x-r, y-r, x+r, y+r, fill=color, outline='white')
            canvas.create_text(x, y-r-12, text=f'{name}{obj.id}', fill='white')
            if name == 'car':
                angle = math.radians(obj.pose.angle)
                canvas.create_line(x, y, x + r*1.8*math.cos(angle), y-r*1.8*math.sin(angle),
                                   fill='white', width=2, arrow='last')
            positions.append((name, obj.id, x, y))
    canvas.create_text(12, 12, anchor='nw', fill='white',
                       text=f"sequence {state.sequence} | {'valid' if state.valid else 'waiting for telemetry'} | +x right, +y up")
    return positions


def plot(argv=None):
    parser = _parser('Display the current provider field, cars and balls')
    args = parser.parse_args(argv)
    import tkinter as tk
    with VSSSProvider.connect(args.endpoint) as client:
        root = tk.Tk()
        root.title(f"VSSS plot — {client.describe()['mode']}")
        canvas = tk.Canvas(root, width=650, height=800, background='#164d32', highlightthickness=0)
        canvas.pack(fill='both', expand=True)
        status = tk.StringVar(value='Waiting for events')
        tk.Label(root, textvariable=status, anchor='w').pack(fill='x')
        lock = Lock()
        latest = [client.gps.state(), '']
        def receive(event):
            with lock:
                if event.topic == 'gps':
                    latest[0] = event.value
                elif event.topic == 'commands':
                    command = event.value
                    latest[1] = f'car{command.car_id}: {command.value} ({command.reason})'
                else:
                    latest[1] = str(event.value)
        with client.subscribe(['gps', 'commands', 'lifecycle'], receive) as subscription:
            def refresh():
                with lock:
                    state, message = latest
                draw_world(canvas, state)
                status.set(str(subscription.error) if subscription.error else message)
                if not subscription.error:
                    root.after(50, refresh)
            root.bind('<Escape>', lambda event: root.destroy())
            root.after(0, refresh)
            try:
                root.mainloop()
            except KeyboardInterrupt:
                pass
            finally:
                try:
                    root.destroy()
                except tk.TclError:
                    pass


def key_velocity(keys, speed):
    return Velocity(speed * (('w' in keys) - ('s' in keys)),
                    speed * (('a' in keys) - ('d' in keys)),
                    speed * (('q' in keys) - ('e' in keys)))


def teleop(argv=None):
    parser = _parser('Hold W/S, A/D, Q/E to control a car; release to stop')
    parser.add_argument('--car', type=int, default=1)
    parser.add_argument('--speed', type=float, default=.3, help='Normalized speed, 0..1 (default .3)')
    args = parser.parse_args(argv)
    if args.car < 0 or not 0 < args.speed <= 1:
        parser.error('--car must be nonnegative and --speed must be in (0, 1]')
    import tkinter as tk
    with VSSSProvider.connect(args.endpoint) as client:
        if not client.describe()['capabilities']['commands']:
            parser.error('playback is read-only')
        root = tk.Tk()
        root.title(f'VSSS teleop — car{args.car}')
        tk.Label(root, padx=30, pady=20, text=(
            f'car{args.car} | speed {args.speed}\n\n'
            'Focus this window and HOLD keys:\n'
            'W / S: forward / backward\nA / D: strafe left / right\n'
            'Q / E: rotate counterclockwise / clockwise\n\n'
            'Release keys or change window: STOP\nSpace: STOP | Escape: exit')).pack()
        status = tk.StringVar(value='Stopped')
        tk.Label(root, textvariable=status, pady=10).pack()
        keys = set()
        previous = [None]
        def press(event):
            key = event.keysym.lower()
            if key == 'space':
                keys.clear()
            elif key in 'wasdqe':
                keys.add(key)
        def release(event):
            keys.discard(event.keysym.lower())
        root.bind('<KeyPress>', press)
        root.bind('<KeyRelease>', release)
        root.bind('<FocusOut>', lambda event: keys.clear())
        root.bind('<Escape>', lambda event: root.destroy())
        def tick():
            velocity = key_velocity(keys, args.speed)
            try:
                # Refresh held motion; idle clients do not keep claiming ownership.
                if velocity != Velocity() or velocity != previous[0]:
                    client.cars.set_velocity(args.car, velocity)
                    previous[0] = velocity
                status.set(str(velocity))
            except Exception as error:
                keys.clear()
                status.set(f'Command failed: {error}. Close and reconnect.')
                return
            root.after(100, tick)
        root.after(0, tick)
        try:
            root.mainloop()
        except KeyboardInterrupt:
            pass
        finally:
            try:
                client.cars.stop(args.car)
            finally:
                # The window may already have been destroyed by its close button.
                try:
                    root.destroy()
                except tk.TclError:
                    pass


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'teleop':
        teleop(sys.argv[2:])
    else:
        plot()
