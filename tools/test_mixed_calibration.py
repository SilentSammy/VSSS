"""Synthetic-plant checks for the two-stage compensation math and firmware."""
import itertools
import json
import sys
import types
import unittest
from unittest.mock import patch

from mp.axis_trim import AxisTrim, mixed_features
from mp.robot_config import validate
from tools._fit_mixed import PAIRS, fit_mixed
from vsss._ble_protocol import normalize_robot_config, CONFIG_MAX_BYTES
from tools.calibrate_staged import VerificationRollback, StageMonitor, burst_durations, steady_yaw_rate
from tools.refine_forward_yaw import secant_coefficient
from vsss.gps import Pose


class MixedCalibrationTests(unittest.TestCase):
    def test_damped_proposal_is_checked_against_firmware_clamp(self):
        rows=[]
        for level in (.25,.5):
            for pair,(a,b,third) in enumerate(PAIRS):
                for sa,sb in itertools.product((-1,1),repeat=2):
                    command=[0.,0.,0.]
                    command[a]=sa*level
                    command[b]=sb*(.1 if b==2 else level)
                    measured=[0.,0.,0.]
                    if pair==0:
                        measured[third]=.8*mixed_features(command)[0]
                    rows.append(dict(command=command,pair=pair,duration_s=1.,
                                     frames=6,response=measured))
        response=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
        with self.assertRaisesRegex(ValueError,'correction clamp'):
            fit_mixed(rows,response)
        proposal=fit_mixed(rows,response,step=.5)
        self.assertAlmostEqual(proposal['mixed_trim'][12],-.4)
        self.assertAlmostEqual(proposal['undamped_mixed_trim'][12],-.8)
        with self.assertRaises(ValueError):
            fit_mixed(rows,response,step=float('nan'))

    def test_steady_yaw_unwrap_and_measured_secant(self):
        samples=[(0.,179.),(.2,-179.),(.4,-177.),(.6,-175.)]
        self.assertAlmostEqual(steady_yaw_rate(samples),10.)
        self.assertAlmostEqual(secant_coefficient(-.12,-10.,-.10,-4.),-.0866666667)
        with self.assertRaises(ValueError):
            secant_coefficient(-.12,1.,-.10,1.1)
    def test_sixty_percent_longer_bursts(self):
        durations=burst_durations(1.6)
        for key,value in dict(forward=1.28,strafe=1.92,yaw=1.92,mixed=1.12).items():
            self.assertAlmostEqual(durations[key],value)
        with self.assertRaises(ValueError):
            burst_durations(float('nan'))
    def test_burst_measurement_uses_host_udp_without_other_gates(self):
        monitor=StageMonitor.__new__(StageMonitor)
        monitor.drain=lambda: None
        monitor.car=types.SimpleNamespace(number=1,is_connected=True)
        pose=Pose(5.,-4.,179.)  # Outside former field limits, by design.
        monitor.gps=types.SimpleNamespace(latest=types.SimpleNamespace(
            usable=True,cars={1:pose,16:Pose(5.1,-4.,0.)}))
        self.assertEqual(monitor.check(),pose)
        monitor.gps.latest.usable=False
        with self.assertRaisesRegex(RuntimeError,'UDP pose unavailable'):
            monitor.check()

    def test_single_axis_and_zero_are_exactly_unchanged(self):
        matrix=(1.,.02,.01, -.03,1.,.04, .01,-.02,1.)
        linear=AxisTrim(matrix)
        mixed=AxisTrim(matrix,mixed=[.2]*18)
        for axis in range(3):
            for level in (-1.,-.5,-.05,0.,.05,.5,1.):
                command=[0.,0.,0.]
                command[axis]=level
                self.assertEqual(linear.apply(command),mixed.apply(command))

    def test_odd_features_distinguish_reversed_diagonals(self):
        v=(.5,-.5,0.)
        self.assertEqual(mixed_features(tuple(-x for x in v)),
                         tuple(-x for x in mixed_features(v)))
        self.assertNotEqual(mixed_features(v),mixed_features((.5,.5,0.)))

    def test_fitted_correction_cancels_synthetic_third_axis_without_active_error(self):
        response=[[.6,.01,.003],[.02,.5,.001],[2.,-3.,200.]]
        slopes=((8.,-6.),(.02,-.01),(.01,.02))
        rows=[]
        for level in (.25,.5):
            for pair,(a,b,third) in enumerate(PAIRS):
                for sa,sb in itertools.product((-1,1),repeat=2):
                    command=[0.,0.,0.]
                    command[a]=sa*level
                    command[b]=sb*(.1 if b==2 else level)
                    feature=mixed_features(command)[pair*2:pair*2+2]
                    residual=sum(x*y for x,y in zip(feature,slopes[pair]))
                    measured=[0.,0.,0.]
                    measured[third]=residual*.8
                    rows.append(dict(command=command,pair=pair,duration_s=.8,
                                     frames=6,response=measured))
        proposal=fit_mixed(rows,response)
        for row in rows:
            feature=mixed_features(row['command'])
            correction=[sum(proposal['mixed_trim'][6*i+j]*feature[j] for j in range(6))
                        for i in range(3)]
            output=[sum(response[i][j]*correction[j] for j in range(3)) for i in range(3)]
            for i in range(3):
                self.assertAlmostEqual(output[i]+row['response'][i]/.8,0.,places=4)
        with self.assertRaises(ValueError):
            fit_mixed(rows[:12],response)  # Only one power level.
        with self.assertRaises(ValueError):
            fit_mixed(rows,[[0.]*3 for _ in range(3)])
        quiet=fit_mixed(rows,response,deadbands=(1.,1.,100.))
        self.assertEqual(quiet['mixed_trim'],[0.]*18)

    def test_schema_migrates_legacy_and_rejects_invalid_coefficients(self):
        config=validate({})
        self.assertEqual(config['mixed_trim'],[0.]*18)
        for key in ('wheel_gains','wheel_reverse_gains','mixed_trim'):
            config.pop(key,None)
        config=normalize_robot_config(config)
        self.assertEqual(config['mixed_trim'],[0.]*18)
        self.assertLess(len(json.dumps(config).encode()),CONFIG_MAX_BYTES)
        for values in ([0.]*17,[float('nan')]*18,[1.01]*18):
            invalid=dict(config,mixed_trim=values)
            with self.assertRaises(ValueError):
                validate(invalid)
            with self.assertRaises(ValueError):
                normalize_robot_config(invalid)

    def test_compensation_is_bounded_and_zero_still_stops(self):
        trim=AxisTrim(mixed=[1.]*18)
        self.assertEqual(trim.apply((0.,0.,0.)),(0.,0.,0.))
        corrected=trim.apply((.5,.5,.5))
        self.assertTrue(all(abs(v-.65)<1e-12 for v in corrected))

    def test_verification_preflight_failure_restores_trial(self):
        class Car:
            stopped=False
            restored=None
            def stop(self):
                self.stopped=True
            def set_trim(self,value):
                self.restored=value
        car=Car()
        with self.assertRaises(RuntimeError):
            with VerificationRollback(car) as transaction:
                transaction.baseline={'trim_matrix':[1.,0.,0.,0.,1.,0.,0.,0.,1.]}
                raise RuntimeError('camera unavailable before first pulse')
        self.assertTrue(car.stopped)
        self.assertEqual(car.restored,[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])

    def test_original_low_speed_pwm_frequency_is_preserved(self):
        class Pin:
            OUT=1
            def __init__(self,*args,**kwargs):
                pass
            def value(self,*args):
                pass
        class PWM:
            def __init__(self,pin,freq):
                self.frequency=freq
                self.duty=None
            def freq(self):
                return self.frequency
            def duty_u16(self,value):
                self.duty=value
            def deinit(self):
                pass
        machine=types.ModuleType('machine')
        machine.Pin,machine.PWM=Pin,PWM
        with patch.dict(sys.modules,{'machine':machine}):
            from mp.mecanum import Motor
            motor=Motor(1,2)
            motor.set_power(.05)
            self.assertEqual(motor.pwm1.freq(),10)
            motor.set_power(.25)
            self.assertEqual(motor.pwm1.freq(),25)
            self.assertEqual(motor.pwm1.duty,int(.25*65535))
            motor.set_power(-.5)
            self.assertIsNone(motor.pwm1)
            self.assertEqual(motor.pwm2.freq(),50)
            motor.stop()
            self.assertIsNone(motor.pwm2)


if __name__=='__main__':
    unittest.main()
