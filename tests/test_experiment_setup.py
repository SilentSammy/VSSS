import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mecanum_client import normalize_trim_matrix
from tests.test_calibration import SimulatedLab
from tests.experiments.recording import Recorder
from tests.experiments.run import main, parse_args
from tests.experiments.plans import build_plan
from tests.experiments.scenarios import CorrectionExperiment, WaypointExperiment
from tests.experiments.observation import observe


class Lab(SimulatedLab):
    def __init__(self):
        super().__init__()
        self.trim_matrix=normalize_trim_matrix()
        self.trim_writes=[]
        self.balls=()
        self.feedback_problem=None

    def get_car(self,number):
        return self.pose if number==1 else None

    def wait_for_position(self):
        return self.pose

    def set_trim(self,matrix=None):
        self.trim_matrix=normalize_trim_matrix(matrix)
        self.trim_writes.append(self.trim_matrix)
        return self.trim_matrix

    def close(self):
        pass


class SetupTests(unittest.TestCase):
    seed=Path(__file__).parent/'experiments/configs/correction_seed.json'

    def test_every_preview_avoids_gps_and_ble(self):
        for arguments in ([],['--experiment','waypoints'],
                          ['--experiment','correction','--trim-file',str(self.seed)]):
            with self.subTest(arguments=arguments), patch('tests.experiments.run.Car') as car, \
                    patch('tests.experiments.run.GPS') as gps, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments),0)
                car.assert_not_called()
                gps.assert_not_called()

    def test_passive_recording_leaves_commands_unknown_and_never_connects_ble(self):
        lab=Lab()
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)/'observe'
            with patch('tests.experiments.run.Car') as car, patch('tests.experiments.run.GPS',return_value=lab) as gps, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--observe','.01','--output',str(directory)]),2)
                car.assert_not_called()
                gps.assert_called_once_with(port=5000,stale_after=.5,subscribe=False)
            self.assertEqual(lab._client._sequence,0)
            self.assertEqual(lab.trim_writes,[])
            rows=list(csv.DictReader((directory/'samples.csv').read_text().splitlines()))
            self.assertTrue(all(row['command_owned']=='False' for row in rows))
            self.assertTrue(all(row['input_x']=='' and row['input_w']=='' for row in rows))

    def test_passive_readiness_detects_trailing_silence(self):
        lab=Lab()
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            directory=Path(temp)
            args=parse_args(['--observe','1','--period','.1'])
            started=lab.time
            def advance(dt):
                lab.time+=dt
                if lab.time-started < .21:
                    lab.publish()
            self.assertEqual(observe(args,directory,lab,clock=lab.now,sleep=advance),2)
            result=json.loads((directory/'result.json').read_text())
            self.assertEqual(result['telemetry_packets'],2)
            self.assertGreater(result['seconds_since_last_packet'],.5)
            self.assertGreater(result['max_receipt_gap_s'],.5)
            self.assertFalse(result['ready'])

    def test_passive_readiness_detects_buffered_stale_frames(self):
        from gps import Feedback
        lab=Lab()
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            directory=Path(temp)
            args=parse_args(['--observe','.7','--period','.1'])
            ticks=0
            def advance(dt):
                nonlocal ticks
                ticks+=1
                lab.time+=dt
                if ticks==3:
                    stale=Feedback(lab.time,True,{1:lab.pose},(),1000.,3,100.,.7,.8,False)
                    for listener in tuple(lab.listeners):listener(stale)
                else:
                    lab.publish()
            self.assertEqual(observe(args,directory,lab,clock=lab.now,sleep=advance),2)
            result=json.loads((directory/'result.json').read_text())
            self.assertEqual(result['unusable_packets'],1)
            self.assertEqual(result['max_network_excess_delay_s'],.7)
            self.assertIn('delayed',result['readiness_reason'])
            rows=list(csv.DictReader((directory/'samples.csv').read_text().splitlines()))
            self.assertTrue(any(r['packet_usable']=='False' and r['pose_valid']=='False' for r in rows))

    def test_simulated_comparison_records_both_matrices(self):
        lab=Lab()
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            directory=Path(temp)
            args=parse_args(['--experiment','correction','--trim-file',str(self.seed),
                             '--duration','.2','--levels','.2','--settle','.1','--center-hold','.1','--heading','0','--center-y','0'])
            log=Recorder(directory,lab,lab,(0.,0.,0.),clock=lab.now)
            runner=CorrectionExperiment(lab,lab,log,args,directory,clock=lab.now,sleep=lab.sleep)
            try:runner.run(build_plan(args))
            finally:log.close()
            self.assertEqual(lab.trim_writes,[normalize_trim_matrix(),args.trim_matrix])
            rows=list(csv.DictReader((directory/'samples.csv').read_text().splitlines()))
            self.assertEqual([r['note'] for r in rows if r['event']=='trim_configuration'],['identity','candidate'])
            self.assertTrue(any(json.loads(r['trim_matrix'])[2][0]<0 for r in rows))

    def test_simulated_waypoints_share_bounds_and_controller(self):
        lab=Lab()
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            directory=Path(temp)
            path=directory/'waypoints.json';path.write_text('[[0.08,0,0],[-0.08,0.08,0]]')
            args=parse_args(['--experiment','waypoints','--waypoints',str(path),'--controller','p',
                             '--settle','.1','--center-hold','.1','--heading','0','--center-timeout','10','--center-y','0'])
            log=Recorder(directory,lab,lab,(0.,0.,0.),clock=lab.now)
            try:WaypointExperiment(lab,lab,log,args,directory,clock=lab.now,sleep=lab.sleep).run(build_plan(args))
            finally:log.close()
            self.assertEqual(json.loads((directory/'result.json').read_text())['status'],'completed')
            rows=list(csv.DictReader((directory/'samples.csv').read_text().splitlines()))
            self.assertTrue(any(r['phase']=='waypoint' and r['controller']=='p' for r in rows))

    def test_comparison_restores_matrix_when_scenario_fails(self):
        lab=Lab();original=lab.trim_matrix
        class Failing:
            def __init__(self,*args):pass
            def run(self,plan):
                lab.set_trim(((1,0,0),(0,1,0),(-.1,0,1)))
                raise RuntimeError('Simulated failure')
        with tempfile.TemporaryDirectory() as temp:
            with patch('tests.experiments.run.Car',return_value=contextlib.nullcontext(lab)), \
                    patch('tests.experiments.run.GPS',return_value=lab), \
                    patch.dict('tests.experiments.run.SCENARIOS',{'correction':Failing}), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'Simulated failure'):
                    main(['--execute','--experiment','correction','--trim-file',str(self.seed),
                          '--keep-trim','--output',str(Path(temp)/'failed')])
            self.assertEqual(lab.trim_matrix,original)

    def test_out_of_bounds_waypoints_are_rejected_before_hardware(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'bad.json';path.write_text('[[0.75,0,0]]')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--experiment','waypoints','--waypoints',str(path)])
