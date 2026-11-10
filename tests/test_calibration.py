import csv
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from tests.experiments.recording import Recorder
from tests.experiments.run import Experiment, Segment, Stride, build_plan, parse_args
from tests.experiments.summarize import summarize
from tests.experiments.controllers import CarController, PredictiveController
from gps import Feedback, Pose


class SimulatedLab:
    """Deterministic kinematic plant with the same pose/command conventions."""
    def __init__(self):
        self.time = 1.
        self.x, self.y, self.angle = .08, -.04, 20.
        self.stale_after = .5
        self.number = 1
        self.is_connected = True
        self.listeners = set()
        self.frame = 0
        self._client = SimpleNamespace(x=0., y=0., w=0., _sequence=0, _written=0, resolution=.05)
        self.publish()

    def now(self):
        return self.time

    @property
    def pose(self):
        return self.latest.cars.get(1)

    def publish(self):
        self.frame += 1
        self.latest = Feedback(self.time, True, {1:Pose(self.x, self.y, self.angle)}, (),
                               self.time+1000, self.frame, 20.)
        for callback in tuple(self.listeners):
            callback(self.latest)

    def sleep(self, dt):
        c = self._client
        theta = math.radians(self.angle)
        self.x += .4*(c.x*math.cos(theta)-c.y*math.sin(theta))*dt
        self.y += .4*(c.x*math.sin(theta)+c.y*math.cos(theta))*dt
        self.angle = (self.angle+100*c.w*dt+180)%360-180
        self.time += dt
        self.publish()

    def set_velocity(self, x=0., y=0., w=0.):
        c = self._client
        c.x, c.y, c.w = (round(v/.05)*.05 for v in (x,y,w))
        c._sequence += 1
        c._written = c._sequence

    def stop(self):
        self.set_velocity()

    def add_listener(self, callback):
        self.listeners.add(callback)

    def remove_listener(self, callback):
        self.listeners.discard(callback)


class CalibrationTests(unittest.TestCase):
    def setup_run(self, directory, lab):
        args = parse_args(['--duration','.2','--settle','.1','--center-hold','.1','--center-y','0',
                           '--center-timeout','8','--levels','.2','--heading','0'])
        log = Recorder(directory, lab, lab, (0.,0.,0.), clock=lab.now)
        runner = Experiment(lab, lab, log, args, directory, clock=lab.now, sleep=lab.sleep)
        return args, log, runner

    def test_full_plan_records_recentering_pulses_and_stops(self):
        lab = SimulatedLab()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            args, log, runner = self.setup_run(directory, lab)
            try:
                runner.run(build_plan(args))
            finally:
                log.close()
            with (directory/'samples.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(json.loads((directory/'result.json').read_text())['status'], 'completed')
            self.assertEqual((lab._client.x, lab._client.y, lab._client.w), (0.,0.,0.))
            for phase in ('baseline','recenter','pre_pulse','pulse','settle','final_rest'):
                self.assertTrue(any(r['phase']==phase and r['event']=='telemetry' for r in rows), phase)
            self.assertTrue(any(r['phase']=='recenter' and float(r['input_x'])!=0 for r in rows))
            self.assertEqual({int(r['trial']) for r in rows if r['phase']=='pulse'}, set(range(1,7)))
            self.assertTrue(all(float(a['t_s'])<=float(b['t_s']) for a,b in zip(rows,rows[1:])))
            self.assertTrue(all(r['frame_sequence'] for r in rows if r['event']=='telemetry'))
            self.assertTrue(any(r['event']=='stop_ack' for r in rows))
            summary = summarize(directory/'samples.csv')
            self.assertEqual(len(summary),6)
            self.assertTrue(all(row['pulse_finished'] for row in summary))
            self.assertGreater(summary[0]['forward_m_s'],0)
            self.assertLess(summary[1]['forward_m_s'],0)

    def test_resume_keeps_original_trial_numbers(self):
        lab=SimulatedLab()
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            args,log,runner=self.setup_run(directory,lab)
            args.start_trial=4
            try:
                runner.run(build_plan(args))
            finally:
                log.close()
            self.assertEqual([row['trial'] for row in summarize(directory/'samples.csv')],[4,5,6])

    def test_faults_stop_and_preserve_partial_recordings(self):
        for fault in ('missing','boundary','operator','disconnect','timeout'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                lab = SimulatedLab()
                args, log, runner = self.setup_run(directory, lab)
                if fault=='missing':
                    lab.latest = Feedback(lab.time,True,{},())
                    args.telemetry_recovery = .01
                elif fault=='boundary':
                    lab.x=1.;lab.publish()
                elif fault=='operator':
                    (directory/'STOP').touch()
                elif fault=='disconnect':
                    lab.is_connected=False
                else:
                    args.center_timeout=.01
                try:
                    with self.assertRaises((RuntimeError, TimeoutError)):
                        runner.run(build_plan(args))
                finally:
                    log.close()
                self.assertEqual((lab._client.x,lab._client.y,lab._client.w),(0.,0.,0.))
                self.assertEqual(json.loads((directory/'result.json').read_text())['status'],'aborted')
                with (directory/'samples.csv').open() as stream:
                    self.assertTrue(any(r['event']=='abort' for r in csv.DictReader(stream)))

    def test_board_to_robot_transform_and_shortest_heading(self):
        lab=SimulatedLab()
        controller=CarController(kd_dist=0,kd_heading=0,time_fn=lab.now)
        cmd=controller.go_to(0,0,math.pi/2,tx=1,ty=0)
        self.assertAlmostEqual(cmd['x'],0,places=6)
        self.assertLess(cmd['y'],0)
        controller.reset()
        cmd=controller.go_to(0,0,math.radians(179),ttheta=math.radians(-179))
        self.assertGreater(cmd['w'],0)
        self.assertLess(cmd['w'],.1)

    def test_predictive_controller_projects_motion_and_respects_limits(self):
        lab=SimulatedLab()
        controller=PredictiveController(kp_dist=2.,kp_heading=1.,max_speed=.3,
                                        max_w=.4,horizon=.3,velocity_alpha=1.,time_fn=lab.now)
        first=controller.go_to(0,0,0,tx=1,ty=0,ttheta=math.pi)
        self.assertAlmostEqual(first['x'],.3)
        self.assertAlmostEqual(first['w'],-.4)  # pi wraps to -pi
        lab.time+=.1
        moving=controller.go_to(.02,0,0,tx=.1,ty=0,ttheta=0)
        self.assertLess(moving['x'],.16)  # projected 6 cm forward before P control
        controller.reset()
        self.assertEqual(controller.velocity,[0.,0.,0.])

    def test_dynamic_segments_do_not_stop_between_axis_changes(self):
        lab=SimulatedLab()
        lab.x=lab.y=lab.angle=0.;lab.publish()
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            args=parse_args(['--profile','dynamic','--stride-scale','.1','--settle','.1',
                             '--center-hold','.1','--heading','0'])
            log=Recorder(directory,lab,lab,(0.,0.,0.),clock=lab.now)
            runner=Experiment(lab,lab,log,args,directory,clock=lab.now,sleep=lab.sleep)
            try:
                runner.run(build_plan(args)[2:4])
            finally:log.close()
            rows=list(csv.DictReader((directory/'samples.csv').open()))
            self.assertEqual({r['segment'] for r in rows if r['phase']=='pulse' and r['event']=='control'},{'1','2','3'})
            self.assertFalse(any(r['event']=='stop_request' and r['phase']=='pulse' for r in rows))
            controls=[r for r in rows if r['event']=='control' and r['phase']=='pulse']
            self.assertTrue(any(float(r['input_x']) and float(r['input_y']) for r in controls))
            self.assertEqual(len(summarize(directory/'samples.csv')),6)

    def test_rectangular_guard_and_predictive_recenter(self):
        lab=SimulatedLab()
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            args,log,runner=self.setup_run(directory,lab)
            args.bounds_x,args.bounds_y=.8,1.5
            try:
                lab.x,lab.y=.1,1.;lab.publish()
                runner.guard()  # Legal along Y, despite being outside the old circular radius.
                self.assertFalse(runner.approaching_boundary(lab.pose))
                runner.velocity=(1.,0.)
                self.assertTrue(runner.approaching_boundary(lab.pose))
                lab.x=.81;lab.publish()
                with self.assertRaisesRegex(RuntimeError,'Hard rectangular'):
                    runner.guard()
            finally:log.close()

    def test_boundary_shortens_stride_then_recenters_without_leaving_bounds(self):
        lab=SimulatedLab()
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            args,log,runner=self.setup_run(directory,lab)
            args.bounds_x,args.bounds_y=.4,1.5
            try:
                runner.run([Stride('toward_edge',(Segment((.9,0.,0.),3.),))])
            finally:log.close()
            result=json.loads((directory/'result.json').read_text())
            self.assertEqual(result['status'],'completed')
            self.assertEqual(result['boundary_shortened_trials'],[1])
            rows=list(csv.DictReader((directory/'samples.csv').open()))
            self.assertTrue(any(r['event']=='boundary_recenter' for r in rows))
            self.assertTrue(all(abs(float(r['x_m']))<=.4 for r in rows if r['pose_valid']=='True'))
            self.assertFalse(summarize(directory/'samples.csv')[0]['pulse_finished'])

    def test_invalid_plan_is_rejected_before_connecting(self):
        for arguments in (['--duration','-1'],['--levels','nan'],['--levels','2'],['--period','0']):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                parse_args(arguments)
