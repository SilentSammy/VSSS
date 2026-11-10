"""Queue telemetry and control events onto one continuously flushed CSV."""
import csv
import json
import queue
from threading import Event, Lock, Thread
import time


FIELDS = ('t_s wall_time_ns event phase trial trial_name segment controller command_owned trim_matrix req_x req_y req_w input_x input_y input_w '
          'sequence ack_sequence connected pose_valid x_m y_m heading_deg '
          'received_t_s pose_age_s minimum_pose_age_s network_excess_delay_s packet_usable server_timestamp frame_sequence frame_age_ms '
          'target_x_m target_y_m target_heading_deg note').split()


class Recorder:
    def __init__(self, directory, car, gps, target, clock=time.monotonic, car_number=None):
        self.car, self.gps, self.target = car, gps, target
        self.number = car.number if car is not None else car_number
        self.controller = 'pid' if car is not None else 'none'
        self.clock = clock
        self.started = clock()
        self.phase, self.trial = 'startup', 0
        self.trial_name, self.segment = '', 0
        self.requested = self.held = (0., 0., 0.) if car is not None else (None, None, None)
        self.sequence = 0 if car is not None else None
        self.error = None
        self._lock = Lock()
        self._queue = queue.SimpleQueue()
        self._file = (directory / 'samples.csv').open('w', newline='', encoding='utf-8')
        self._writer = Thread(target=self._write, name='CalibrationCSV', daemon=True)
        self._writer.start()
        self.gps.add_listener(self.telemetry)
        self.record('startup')

    def _write(self):
        try:
            writer = csv.DictWriter(self._file, fieldnames=FIELDS)
            writer.writeheader()
            while True:
                row = self._queue.get()
                if row is None:
                    break
                if isinstance(row, Event):
                    row.set()
                    continue
                writer.writerow(row)
                self._file.flush()  # Progress and partial/aborted runs remain readable.
        except Exception as error:
            self.error = error
        finally:
            self._file.close()

    def record(self, event, feedback=None, note='', requested=None, held=None, sequence=None):
        with self._lock:
            now = self.clock()
            if requested is not None:
                self.requested = requested
            if held is not None:
                self.held = held
            if sequence is not None:
                self.sequence = sequence
            feedback = self.gps.latest if feedback is None else feedback
            pose = None if feedback is None else feedback.cars.get(self.number)
            age = None if feedback is None else now - feedback.received_at
            row = dict(t_s=now-self.started, wall_time_ns=time.time_ns(), event=event,
                       phase=self.phase, trial=self.trial, note=note,
                       trial_name=self.trial_name, segment=self.segment,
                       controller=self.controller, command_owned=self.car is not None,
                       trim_matrix=json.dumps(getattr(self.car,'trim_matrix',None)),
                       sequence=self.sequence, ack_sequence=None if self.car is None else self.car._client._written,
                       connected=None if self.car is None else self.car.is_connected,
                       pose_valid=(pose is not None and feedback.usable and
                                   feedback.minimum_age_at_receipt_s + age <= self.gps.stale_after),
                       target_x_m=self.target[0], target_y_m=self.target[1],
                       target_heading_deg=self.target[2])
            row.update(zip(('req_x', 'req_y', 'req_w'), self.requested))
            row.update(zip(('input_x', 'input_y', 'input_w'), self.held))
            if feedback is not None:
                row.update(received_t_s=feedback.received_at-self.started, pose_age_s=age,
                           minimum_pose_age_s=feedback.minimum_age_at_receipt_s+age,
                           network_excess_delay_s=feedback.network_excess_delay_s,
                           packet_usable=feedback.usable,
                           server_timestamp=feedback.server_timestamp,
                           frame_sequence=feedback.frame_sequence, frame_age_ms=feedback.frame_age_ms)
            if pose is not None:
                row.update(x_m=pose.x, y_m=pose.y, heading_deg=pose.angle)
            self._queue.put(row)

    def telemetry(self, feedback):
        self.record('telemetry', feedback)

    def set_phase(self, phase, trial):
        with self._lock:
            self.phase, self.trial = phase, trial
        self.record('phase')
        print(f'[{trial}] {phase}', flush=True)

    def set_segment(self, trial_name, segment):
        with self._lock:
            self.trial_name, self.segment = trial_name, segment
        self.record('segment')

    def set_target(self, target):
        with self._lock:
            self.target = tuple(target)
        self.record('target')

    def close(self):
        self.gps.remove_listener(self.telemetry)
        self.record('closed')
        self._queue.put(None)
        self._writer.join(timeout=5)
        if self._writer.is_alive():
            raise RuntimeError('CSV writer did not finish')
        if self.error:
            raise RuntimeError(f'CSV recording failed: {self.error}')

    def flush(self, timeout=5):
        """Wait until all rows queued before this call are on disk."""
        marker = Event()
        self._queue.put(marker)
        if not marker.wait(timeout):
            raise RuntimeError('CSV writer did not flush')
        if self.error:
            raise RuntimeError(f'CSV recording failed: {self.error}')
