"""Bounded, buffered navigation flight recorder. Never raises into control."""
try:
    import ujson as json
except ImportError:
    import json
import gc
import math
import os
import struct
import time


COLUMNS = ('t_ms', 'loop_ms', 'mode', 'fault', 'flags', 'waypoint_id',
           'target_x', 'target_y', 'target_deg', 'pose_x', 'pose_y', 'pose_deg',
           'rx_age_ms', 'frame_age_ms', 'frame_sequence', 'packets', 'parse_errors',
           'control_dt_ms', 'velocity_x', 'velocity_y', 'velocity_w_rad_s',
           'command_x', 'command_y', 'command_w', 'trimmed_x', 'trimmed_y', 'trimmed_w',
           'motor_x', 'motor_y', 'motor_w', 'fl_hz', 'fl_duty', 'fr_hz', 'fr_duty',
           'bl_hz', 'bl_duty', 'br_hz', 'br_duty', 'heap_free', 'last_flush_ms')
SAMPLE_FORMAT = '<6i6fif4i12f10i'
SAMPLE_SIZE = struct.calcsize(SAMPLE_FORMAT)
FLOAT_FIELDS = tuple(list(range(6, 12)) + [13] + list(range(18, 30)))
MISSING_INT = -2147483648


def ticks_ms():
    ticks = getattr(time, 'ticks_ms', None)
    return ticks() if ticks else int(time.monotonic() * 1000)


def ticks_diff(now, then):
    diff = getattr(time, 'ticks_diff', None)
    return diff(now, then) if diff else now - then


class NavigationLog:
    """Three 128 KiB JSONL segments, 5 Hz samples only during waypoints.

    navigation.jsonl is newest; .1.jsonl and .2.jsonl are older. Every segment
    has a schema/configuration header. Append across boots, never truncate on
    startup. A preallocated ring retains up to 128 samples (about 25 seconds),
    reduced if needed to leave 20 KiB of free heap for control and networking.
    No filesystem writes occur during active navigation: persist only once
    stopped, reached, faulted, or shutting down. Power loss during motion can
    lose the current RAM trace; earlier flushed waypoints remain on flash.
    """
    def __init__(self, settings, prefix='navigation', max_bytes=128 * 1024,
                 sample_ms=200, buffer_bytes=2048, sample_slots=128):
        self.prefix, self.max_bytes = prefix, max_bytes
        self.sample_ms, self.buffer_bytes = sample_ms, buffer_bytes
        self.enabled = True
        self.buffer, self.pending, self.size = [], 0, 0
        self.file = None
        self.last_sample = self.last_flush = ticks_ms()
        self.last_flush_ms = 0
        self.last_key = None
        self.sample_slots = sample_slots
        self.head = self.count = self.samples_dropped = self.events_dropped = 0
        self.header = dict(type='schema', version=1, columns=COLUMNS,
                           boot_id=os.urandom(4).hex(), car_id=settings['car_id'],
                           controller=settings['controller'], trim=settings['trim_matrix'],
                           sample_ms=sample_ms, sample_slots=sample_slots,
                           persistence='when_stopped', pwm_duty_scale=65535)
        try:
            gc.collect()
            if hasattr(gc, 'mem_free'):
                self.sample_slots = min(sample_slots, max(0, (gc.mem_free() - 20 * 1024) // SAMPLE_SIZE))
            while self.sample_slots >= min(16, sample_slots) and self.sample_slots > 0:
                try:
                    self.ring = bytearray(self.sample_slots * SAMPLE_SIZE)
                    break
                except MemoryError:
                    self.sample_slots //= 2
            else:
                raise MemoryError('insufficient spare heap for navigation log')
            self.header['sample_slots'] = self.sample_slots
            try:
                import machine
                self.header['reset_cause'] = machine.reset_cause()
            except ImportError:
                pass
            path = self.path(0)
            try:
                self.size = os.stat(path)[6]
            except OSError:
                pass
            self.file = open(path, 'a')
            # A newline fences off a torn final JSON row from a power loss.
            self._write('\n' + json.dumps(self.header) + '\n')
            self.event('boot', car_id=settings['car_id'])
            self.flush()
            print('Navigation log: %d samples, %d bytes RAM' % (self.sample_slots, len(self.ring)))
        except Exception as error:
            self._disable(error)

    def path(self, generation):
        return self.prefix + ('.%d' % generation if generation else '') + '.jsonl'

    def _disable(self, error):
        self.enabled = False
        self.ring = None
        self.buffer, self.pending = [], 0
        try:
            if self.file:
                self.file.close()
        except Exception:
            pass
        self.file = None
        print('Navigation logging disabled:', error)

    def _rotate(self):
        self.file.close()
        try:
            os.remove(self.path(2))
        except OSError:
            pass
        for generation in (1, 0):
            try:
                os.stat(self.path(generation))
            except OSError:
                continue
            os.rename(self.path(generation), self.path(generation + 1))
        self.file = open(self.path(0), 'w')
        self.size = 0
        header = json.dumps(self.header) + '\n'
        self.file.write(header)
        self.size = len(header)

    def _write(self, line):
        if self.size + len(line) > self.max_bytes:
            self._rotate()
        if self.size + len(line) > self.max_bytes:
            raise ValueError('navigation log record exceeds segment size')
        self.file.write(line)
        self.size += len(line)

    def _append(self, record):
        line = json.dumps(record) + '\n'
        while self.buffer and self.pending + len(line) > self.buffer_bytes:
            self.pending -= len(self.buffer.pop(0)[1])
            self.events_dropped += 1
        if len(line) > self.buffer_bytes:
            self.events_dropped += 1
            return
        self.buffer.append((record['t_ms'], line))
        self.pending += len(line)

    def event(self, name, **values):
        if self.enabled:
            try:
                self._append(dict(type=name, t_ms=ticks_ms(), **values))
            except Exception as error:
                self._disable(error)

    def configuration(self, trim, controller):
        self.header['trim'], self.header['controller'] = trim, controller
        self.event('configuration', trim=trim, controller=controller)

    def sample(self, now, state, receiver, controller, car, wifi, loop_ms):
        if not self.enabled:
            return
        try:
            key = state['mode'], state['fault'], state['waypoint_id']
            changed = key != self.last_key
            if changed:
                self.last_key = key
                self.event('state', mode=state['mode'], fault=state['fault'],
                           waypoint_id=state['waypoint_id'], target=state['waypoint'])
            if state['mode'] == 2 or changed:
                if changed or ticks_diff(now, self.last_sample) >= self.sample_ms:
                    self.last_sample = now
                    pose = receiver.pose if receiver else None
                    received = receiver.received_ms if receiver else None
                    flags = int(bool(wifi)) | (2 if receiver and receiver.fresh else 0)
                    values = [now, loop_ms, state['mode'], state['fault'], flags, state['waypoint_id']]
                    values.extend(state['waypoint'] or (None, None, None))
                    values.extend(pose or (None, None, None))
                    values.extend((max(0, ticks_diff(now, received)) if received is not None else None,
                                   receiver.frame_age_ms if receiver else None,
                                   receiver.frame_sequence if receiver else None,
                                   receiver.packets if receiver else 0,
                                   receiver.parse_errors if receiver else 0,
                                   state['control_dt_ms']))
                    values.extend(controller.velocity)
                    values.extend(state['control_raw'])
                    values.extend(state['control_command'])
                    values.extend((car.x, car.y, car.w))
                    for motor in (car.motor_fl, car.motor_fr, car.motor_bl, car.motor_br):
                        pwm = motor.pwm1 or motor.pwm2
                        values.extend((pwm.freq() if pwm else 0,
                                       (1 if motor.pwm1 else -1) * pwm.duty_u16() if pwm else 0))
                    values.extend((gc.mem_free() if hasattr(gc, 'mem_free') else None, self.last_flush_ms))
                    for index in range(len(values)):
                        if values[index] is None:
                            values[index] = float('nan') if index in FLOAT_FIELDS else MISSING_INT
                    struct.pack_into(SAMPLE_FORMAT, self.ring, self.head * SAMPLE_SIZE, *values)
                    self.head = (self.head + 1) % self.sample_slots
                    if self.count == self.sample_slots:
                        self.samples_dropped += 1
                    else:
                        self.count += 1
            if changed and state['mode'] != 2 and not any((car.x, car.y, car.w)):
                self.flush()  # Motors are already stopped on reached/fault/idle.
        except Exception as error:
            self._disable(error)

    def flush(self):
        if self.enabled and (self.pending or self.count):
            try:
                started = ticks_ms()
                event_index = 0
                first = (self.head - self.count) % self.sample_slots
                for index in range(self.count):
                    offset = ((first + index) % self.sample_slots) * SAMPLE_SIZE
                    values = list(struct.unpack_from(SAMPLE_FORMAT, self.ring, offset))
                    while event_index < len(self.buffer) and ticks_diff(self.buffer[event_index][0], values[0]) <= 0:
                        self._write(self.buffer[event_index][1])
                        event_index += 1
                    for field in range(len(values)):
                        if ((field in FLOAT_FIELDS and not math.isfinite(values[field])) or
                                (field not in FLOAT_FIELDS and values[field] == MISSING_INT)):
                            values[field] = None
                    self._write(json.dumps(dict(type='sample', v=values)) + '\n')
                for _, line in self.buffer[event_index:]:
                    self._write(line)
                self._write(json.dumps(dict(type='flush', t_ms=ticks_ms(), samples=self.count,
                                            samples_dropped=self.samples_dropped,
                                            events_dropped=self.events_dropped)) + '\n')
                self.file.flush()
                self.buffer, self.pending = [], 0
                self.head = self.count = self.samples_dropped = self.events_dropped = 0
                self.last_flush = ticks_ms()
                self.last_flush_ms = ticks_diff(self.last_flush, started)
            except Exception as error:
                self._disable(error)

    def close(self):
        self.flush()
        if self.file:
            try:
                self.file.close()
            except Exception as error:
                self._disable(error)
            self.file = None
        self.enabled = False
