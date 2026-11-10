"""Reusable motion session: recording, recentering, bounds and stop handling."""
from contextlib import contextmanager
import math
import json
import time
from mecanum_client import to_byte
from .controllers import make_controller, wrap_angle
from .safety import SafetyPolicy


class Experiment:
    def __init__(self, car, gps, recorder, args, directory,
                 clock=time.monotonic, sleep=time.sleep, controller=None):
        self.car, self.gps, self.log = car, gps, recorder
        self.args, self.directory = args, directory
        self.clock, self.sleep = clock, sleep
        self.controller = controller or make_controller(args, clock)
        self.log.controller = args.controller if controller is None else type(controller).__name__
        self.home_target = tuple(recorder.target)
        self.safety = SafetyPolicy(args)
        self.last_status = -math.inf
        self.started = clock()
        self.last_motion_sample = None
        self.velocity = (0., 0.)
        self.truncated_trials = []
        self.completed = False

    def guard(self):
        if (self.directory / 'STOP').exists():
            raise RuntimeError('Operator STOP file detected')
        if self.log.error:
            raise RuntimeError(f'CSV writer failed: {self.log.error}')
        if not self.car.is_connected:
            raise RuntimeError('BLE connection lost')
        pose = self.car.pose
        if pose is None:
            if self.log.phase == 'pulse':
                raise RuntimeError('Camera position missing or stale during active pulse')
            pose = self.recover_telemetry()
        self.safety.check(pose)
        received = self.gps.latest.received_at
        if self.last_motion_sample is not None:
            previous_time, px, py = self.last_motion_sample
            dt = received-previous_time
            if 0 < dt <= self.gps.stale_after:
                self.velocity = tuple(.5*new+.5*old for new,old in zip(
                    ((pose.x-px)/dt, (pose.y-py)/dt), self.velocity))
        if self.last_motion_sample is None or received != self.last_motion_sample[0]:
            self.last_motion_sample = (received, pose.x, pose.y)
        if self.clock() - self.started > self.args.max_seconds:
            raise RuntimeError('Experiment time limit exceeded')
        if self.clock() - self.last_status >= 2:
            status = dict(phase=self.log.phase, trial=self.log.trial,
                          elapsed_s=round(self.clock()-self.started, 2),
                          pose=dict(x=pose.x, y=pose.y, angle=pose.angle))
            temporary = self.directory / 'status.tmp'
            temporary.write_text(json.dumps(status, indent=2))
            temporary.replace(self.directory / 'status.json')
            print(f'  {status["elapsed_s"]:.1f}s | {pose.x:+.3f}, {pose.y:+.3f} m | {pose.angle:+.0f} deg', flush=True)
            self.last_status = self.clock()
        return pose

    def recover_telemetry(self):
        """Stop and reacquire only in phases where repeating no motion is safe."""
        self.log.record('telemetry_pause', note='stopping while camera telemetry recovers')
        self.car.stop()
        deadline = self.clock() + self.args.telemetry_recovery
        while self.clock() < deadline:
            if (self.directory / 'STOP').exists():
                raise RuntimeError('Operator STOP file detected')
            if not self.car.is_connected:
                raise RuntimeError('BLE connection lost while telemetry was unavailable')
            pose = self.car.pose
            if pose is not None:
                self.safety.check(pose)
                self.controller.reset()
                self.last_motion_sample = None
                self.velocity = (0., 0.)
                self.log.record('telemetry_recovered')
                print('  Camera telemetry recovered; resuming from stopped state.', flush=True)
                return pose
            self.sleep(self.args.period)
        raise RuntimeError('Camera position remained stale after stopped recovery window')

    def approaching_boundary(self, pose):
        return self.safety.approaching(pose, self.velocity)

    def before_trial(self, trial):
        return False

    def command(self, values):
        self.log.record('command_request', requested=values)
        self.car.set_velocity(*values)
        client = self.car._client
        wire = tuple(to_byte(v) for v in (client.x, client.y, client.w))
        held = tuple((v if v < 128 else v-256)/128 for v in wire)
        self.log.record('control', held=held, sequence=client._sequence)

    def stop(self):
        if not self.car.is_connected:
            self.log.record('stop_unavailable', requested=(0., 0., 0.),
                            note='BLE disconnected; firmware heartbeat timeout enforces stop')
            self.car.stop()
            return
        # Log the zero setpoint before waiting for the forced stop ACK.
        self.command((0., 0., 0.))
        self.log.record('stop_request')
        self.car.stop()
        self.log.record('stop_ack', sequence=self.car._client._sequence)

    def rest(self, seconds, phase, trial):
        self.log.set_phase(phase, trial)
        self.stop()
        deadline = self.clock()+seconds
        while self.clock() < deadline:
            self.guard()
            self.log.record('control')
            self.sleep(self.args.period)

    def recenter(self, trial):
        return self.move_to(self.home_target, trial, 'recenter')

    def move_to(self, target, trial, phase='waypoint'):
        self.log.set_target(target)
        self.log.set_phase(phase, trial)
        self.controller.reset()
        deadline = self.clock()+self.args.center_timeout
        stable_since = None
        last_received = None
        command = (0., 0., 0.)
        while self.clock() < deadline:
            pose = self.guard()
            position_ok = math.hypot(pose.x-target[0], pose.y-target[1]) <= self.args.position_tolerance
            heading_ok = abs(wrap_angle(math.radians(target[2]-pose.angle))) <= math.radians(self.args.heading_tolerance)
            if position_ok and heading_ok:
                command = (0., 0., 0.)
                if stable_since is None:
                    stable_since = self.clock()
                if self.clock()-stable_since >= self.args.center_hold:
                    self.stop()
                    return
            else:
                stable_since = None
                feedback = self.gps.latest
                # Run PIDs on fresh frames, rather than differentiating repeated poses.
                if feedback.received_at != last_received:
                    cmd = self.controller.go_to(
                        pose.x, pose.y, math.radians(pose.angle),
                        tx=None if position_ok else target[0],
                        ty=None if position_ok else target[1],
                        ttheta=None if heading_ok else math.radians(target[2]))
                    speed = math.hypot(cmd['x'], cmd['y'])
                    if not position_ok and 0 < speed < self.args.min_center_speed:
                        scale = self.args.min_center_speed / speed
                        cmd['x'] *= scale
                        cmd['y'] *= scale
                    if not heading_ok and 0 < abs(cmd['w']) < self.args.min_center_w:
                        cmd['w'] = math.copysign(self.args.min_center_w, cmd['w'])
                    command = (cmd['x'], cmd['y'], cmd['w'])
                    last_received = feedback.received_at
            self.command(command)
            self.sleep(self.args.period)
        raise TimeoutError('Recentering did not converge; inspect the CSV before raising limits')

    def run(self, plan):
        with self.recorded_run():
            self.rest(self.args.settle, 'baseline', 0)
            for trial, pulse in enumerate(plan, 1):
                if trial < self.args.start_trial:
                    continue
                force_center = self.before_trial(trial)
                if force_center or (trial-self.args.start_trial) % self.args.recenter_every == 0:
                    self.recenter(trial)
                self.rest(self.args.settle, 'pre_pulse', trial)
                print(f'Stride {trial}/{len(plan)}: {pulse.name}', flush=True)
                self.log.set_phase('pulse', trial)
                truncated = False
                for segment_id, segment in enumerate(pulse.segments, 1):
                    self.log.set_segment(pulse.name, segment_id)
                    print(f'  segment {segment_id}: {segment.command} for {segment.seconds:.2f}s', flush=True)
                    deadline = self.clock()+segment.seconds
                    while self.clock() < deadline:
                        pose = self.guard()
                        if self.approaching_boundary(pose):
                            self.log.record('boundary_recenter', note='Inner/predicted boundary reached; truncating stride')
                            self.truncated_trials.append(trial)
                            print('  Approaching inner boundary; stopping and recentering.', flush=True)
                            truncated = True
                            break
                        self.command(segment.command)
                        self.sleep(self.args.period)
                    if truncated:
                        break
                self.rest(self.args.settle, 'settle', trial)
                if truncated:
                    self.recenter(trial)
            self.recenter(len(plan)+1)
            self.rest(self.args.settle, 'final_rest', len(plan)+1)
    @contextmanager
    def recorded_run(self):
        outcome = 'completed'
        try:
            yield
        except BaseException as error:
            outcome = 'aborted'
            self.log.set_phase('aborted', self.log.trial)
            self.log.record('abort', note=f'{type(error).__name__}: {error}')
            raise
        finally:
            self.completed = outcome == 'completed'
            try:
                self.stop()
            finally:
                self.log.set_phase(outcome, self.log.trial)
                result = dict(status=outcome, phase=outcome, trial=self.log.trial,
                              boundary_shortened_trials=self.truncated_trials,
                              elapsed_s=self.clock()-self.started)
                (self.directory / 'result.json').write_text(json.dumps(result, indent=2))
                (self.directory / 'status.json').write_text(json.dumps(result, indent=2))


