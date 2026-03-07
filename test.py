"""
Multi-joystick PoC
------------------
- Polls inputs.devices.gamepads every second for new/removed devices.
- Spawns one reader thread per gamepad; each thread prints events tagged with
  the device index so you can tell which physical pad produced each input.
- Press Ctrl+C to exit.
"""

import inputs
import threading
import time

# device_index -> Thread
_threads: dict[int, threading.Thread] = {}
_stop_flags: dict[int, threading.Event] = {}
_lock = threading.Lock()


def _reader(device, index: int, stop: threading.Event):
    print(f"[pad {index}] Started listening: {device.name}")
    while not stop.is_set():
        try:
            events = device.read()
        except Exception as e:
            print(f"[pad {index}] Disconnected ({e}), stopping thread.")
            break
        for ev in events:
            if ev.ev_type in ("Key", "Absolute"):
                print(f"[pad {index}] {ev.ev_type:8s}  {ev.code:20s}  {ev.state}")
    print(f"[pad {index}] Thread exited.")


def _sync_devices():
    """Start threads for new gamepads; clean up threads for gone ones."""
    current_pads = inputs.devices.gamepads          # re-query every call
    current_indices = set(range(len(current_pads)))

    with _lock:
        # --- add new ---
        for i, pad in enumerate(current_pads):
            if i not in _threads or not _threads[i].is_alive():
                stop = threading.Event()
                t = threading.Thread(target=_reader, args=(pad, i, stop), daemon=True)
                _stop_flags[i] = stop
                _threads[i] = t
                t.start()

        # --- remove stale ---
        stale = set(_threads.keys()) - current_indices
        for i in stale:
            _stop_flags[i].set()
            del _threads[i]
            del _stop_flags[i]
            print(f"[pad {i}] Removed (no longer enumerated).")


if __name__ == "__main__":
    print("Multi-joystick PoC — plug/unplug gamepads, press Ctrl+C to quit.\n")
    try:
        while True:
            _sync_devices()
            n = len(_threads)
            if n == 0:
                print("No gamepads detected. Waiting...")
            else:
                print(f"Tracking {n} gamepad(s). Listening for events...")
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nExiting.")
