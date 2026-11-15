"""Hardware-free checks of the current VSSS API and live supervisor."""
import ast
import json
from pathlib import Path
import tempfile
from threading import Event
import time
import unittest
from unittest.mock import patch

from api_registry import schema, SOURCES
from fake_devices import Devices
from live_runtime import LiveRunner
from server import validate

ROOT=Path(__file__).resolve().parent.parent


def program(body):
    return 'from vsss import Car, gps, run, input\nfrom time import sleep, monotonic\n@run\ndef main():\n' + ''.join('    '+line+'\n' for line in body.splitlines()) + '\nif __name__ == "__main__":\n    main()\n'


class LiveTests(unittest.TestCase):
    def setUp(self):
        self.devices=Devices()
        self.runner=self.devices.runner(lease_seconds=2,max_seconds=8)

    def tearDown(self):
        self.runner.shutdown()

    def start(self,body):
        code=program(body)
        validate(code)
        return self.runner.start(code)['run_id']

    def finish(self,run,renew=True,timeout=5):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            result=self.runner.status(run if renew else None)
            if result['state'] not in {'running','stopping'}: return result
            time.sleep(.02)
        self.fail('Run did not finish: '+str(self.runner.status()))

    def wait_for(self,predicate,timeout=3):
        deadline=time.monotonic()+timeout
        while not predicate():
            if time.monotonic()>deadline: self.fail('Condition did not become true')
            time.sleep(.01)

    def assert_closed(self):
        self.assertTrue(self.devices.cars)
        for car in self.devices.cars:
            self.assertIn(('stop',),car.events)
            self.assertIn(('close',),car.events)

    def test_multiple_cars_and_two_native_gamepads_in_one_loop_without_gps(self):
        run=self.start('a = Car(1)\nb = Car(2)\nfor i in range(3):\n    a.set_velocity(x=input.get_axis("LY0"))\n    b.set_velocity(x=input.get_axis("LY1"))\n    sleep(.01)\nprint(input.is_pressed("A0"), input.is_pressed("B1"))')
        result=self.finish(run)
        self.assertEqual(result['state'],'finished',result)
        self.assertIn('True True',result['output'])
        self.assertEqual([e[1] for e in self.devices.cars[0].events if e[0]=='velocity'],[.5]*3)
        self.assertEqual([e[1] for e in self.devices.cars[1].events if e[0]=='velocity'],[-.5]*3)
        self.assertEqual(self.devices.field_reads,0)
        self.assertEqual(self.devices.input_reads,1)
        self.assert_closed()

    def test_keyboard_edges_and_bipolar_controls_use_actual_api_functions(self):
        run=self.start('print(input.is_pressed("w"))\nprint(input.rising_edge("w"), input.rising_edge("w"))\nprint(input.falling_edge("s"), input.falling_edge("s"))\nprint(input.get_bipolar_ctrl(high_key="w", low_key="s", keyboard_scale=.3))\nprint(input.get_axis("LY1"))')
        result=self.finish(run)
        self.assertEqual(result['state'],'finished',result)
        self.assertEqual(result['output'],'True\nTrue False\nTrue False\n0.3\n-0.5\n')
        self.assertFalse(self.devices.cars)
        self.assertEqual(self.devices.field_reads,0)

    def test_iterating_a_list_of_car_objects(self):
        run=self.start('fleet=[Car(1),Car(2),Car(3)]\nfor tick in range(2):\n    for robot in fleet:\n        robot.set_velocity(x=.1*robot.number)\n    sleep(.01)')
        result=self.finish(run)
        self.assertEqual(result['state'],'finished',result)
        self.assertEqual(len(self.devices.cars),3)
        for car in self.devices.cars:
            values=[event[1] for event in car.events if event[0]=='velocity']
            self.assertEqual(len(values),2)
            self.assertAlmostEqual(values[0],.1*car.number)
        self.assert_closed()

    def test_field_without_car_and_lossless_pose_status_records(self):
        run=self.start('cars = gps.cars\nballs = gps.balls\nprint(cars[1].x, cars.get(2).angle, balls[0].y)\nprint(gps.get_car(9) is None, gps.board_detected)')
        result=self.finish(run)
        self.assertEqual(result['state'],'finished',result)
        self.assertEqual(result['output'],'0.1 90.0 0.4\nTrue True\n')
        self.assertFalse(self.devices.cars)
        self.assertEqual(self.devices.field_reads,1)
        self.assertGreater(self.devices.gps.closes,0)

    def test_current_boilerplates_execute_unchanged(self):
        for name in ('velocity',):
            with self.subTest(name=name):
                code=(ROOT/f'boilerplate_{name}.py').read_text(encoding='utf-8-sig')
                validate(code)
                result=self.finish(self.runner.start(code)['run_id'])
                self.assertEqual(result['state'],'finished',result)
                self.assert_closed()
        code=(ROOT/'boilerplate_observe.py').read_text(encoding='utf-8-sig')
        validate(code)
        run=self.runner.start(code)['run_id']
        self.wait_for(lambda:'Balls:' in self.runner.status(run)['output'])
        self.runner.stop(run)
        self.assertEqual(self.finish(run)['state'],'stopped')

    def test_readable_status_and_pose(self):
        run=self.start('car=Car(1)\nprint(car.number)\nprint(car.wait_for_position(timeout=.1).angle)\ncar.go_to(0,.5,90)\ns=car.read_status()\nprint(s.mode_name,s.fault_name,s.pose_fresh)')
        result=self.finish(run)
        self.assertEqual(result['state'],'finished',result)
        self.assertIn('reached none True',result['output'])
        self.assertIn(('waypoint',0,.5,90),self.devices.cars[0].events)

    def test_stop_interrupts_wait_and_cpu_loop_for_all_cars(self):
        for tail in ('sleep(60)','while True:\n    pass'):
            with self.subTest(tail=tail):
                self.devices.cars.clear()
                run=self.start('a=Car(1)\nb=Car(2)\na.set_velocity(.2)\nb.set_velocity(.3)\n'+tail+'\na.go_to(0,.5,90)')
                self.wait_for(lambda:len(self.devices.cars)==2 and any(e[0]=='velocity' for e in self.devices.cars[1].events))
                self.runner.stop(run)
                self.assertEqual(self.finish(run)['state'],'stopped')
                self.assert_closed()
                self.assertFalse(any(e[0]=='waypoint' for c in self.devices.cars for e in c.events))

    def test_error_stops_all_and_preserves_original_error(self):
        run=self.start('a=Car(1)\nb=Car(2)\na.set_velocity(.2)\nb.set_velocity(.2)\nraise RuntimeError("student failure")')
        result=self.finish(run)
        self.assertEqual(result['state'],'failed')
        self.assertIn('student failure',result['error'])
        self.assert_closed()

    def test_stop_other_car_while_second_connection_is_blocked(self):
        entered,release=Event(),Event()
        def factory(number,**kwargs):
            if number==2:
                entered.set(); release.wait(3)
            return self.devices.car(number,**kwargs)
        self.runner=LiveRunner(factory,gps_factory=self.devices.field,input_factory=self.devices.inputs,cleanup=self.devices.cleanup)
        run=self.start('a=Car(1)\na.set_velocity(.2)\nb=Car(2)\nb.set_velocity(.2)')
        self.assertTrue(entered.wait(2))
        self.runner.stop(run)
        self.wait_for(lambda:('close',) in self.devices.cars[0].events,timeout=1)
        release.set()
        self.assertEqual(self.finish(run)['state'],'stopped')
        self.assertFalse(any(e[0]=='velocity' for e in self.devices.cars[1].events))
        self.assert_closed()

    def test_failed_second_connection_and_failed_first_cleanup_still_stop_others(self):
        def factory(number,**kwargs):
            if number==2: raise TimeoutError('second car missing')
            return self.devices.car(number,**kwargs)
        self.runner=LiveRunner(factory,cleanup=lambda:None)
        result=self.finish(self.start('a=Car(1)\na.set_velocity(.2)\nb=Car(2)'))
        self.assertIn('second car missing',result['error'])
        self.assert_closed()
        self.devices.cars.clear()
        def broken_stop(number,**kwargs):
            car=self.devices.car(number,**kwargs)
            if number==1:
                def stop(): raise TimeoutError('no stop acknowledgement')
                car.stop=stop
            return car
        self.runner=LiveRunner(broken_stop,cleanup=lambda:None)
        result=self.finish(self.start('a=Car(1)\nb=Car(2)\nraise ValueError("original")'))
        self.assertIn('original',result['error'])
        self.assertIn('no stop acknowledgement',result['error'])
        for car in self.devices.cars: self.assertIn(('close',),car.events)
        self.assertIn(('stop',),self.devices.cars[1].events)

    def test_disconnect_any_car_stops_all(self):
        run=self.start('a=Car(1)\nb=Car(2)\na.set_velocity(.2)\nb.set_velocity(.2)\nsleep(60)')
        self.wait_for(lambda:len(self.devices.cars)==2 and any(e[0]=='velocity' for e in self.devices.cars[1].events))
        self.devices.cars[1].connection_generation+=1
        result=self.finish(run)
        self.assertEqual(result['state'],'stopped')
        self.assertIn('car2',result['reason'])
        self.assert_closed()

    def test_gps_is_optional_for_velocity_even_when_pose_missing(self):
        def factory(number,**kwargs):
            car=self.devices.car(number,**kwargs); car.pose=None; return car
        self.runner=LiveRunner(factory,gps_factory=lambda: self.fail('GPS must remain lazy'),cleanup=lambda:None)
        result=self.finish(self.start('a=Car(1)\na.set_velocity(.2)\nsleep(.05)'))
        self.assertEqual(result['state'],'finished',result)
        self.assert_closed()

    def test_lease_limit_busy_and_stale_stop(self):
        self.runner=self.devices.runner(lease_seconds=.3,max_seconds=2)
        run=self.start('a=Car(1)\nb=Car(2)\nsleep(60)')
        with self.assertRaises(ValueError): self.runner.start('print(1)')
        self.runner.stop('old-run')
        self.assertEqual(self.runner.status()['state'],'running')
        result=self.finish(run,renew=False)
        self.assertIn('Editor disconnected',result['reason'])
        self.assert_closed()
        self.runner=self.devices.runner(lease_seconds=2,max_seconds=.3)
        result=self.finish(self.start('while True:\n    pass'))
        self.assertIn('time limit',result['reason'])

    def test_instructor_operations_are_denied_by_validator_and_runner(self):
        forbidden = ['a.set_trim()', 'a.read_trim()', 'a.set_controller({"horizon":.2})',
                     'a.read_controller()', 'a.restart()', 'a.reconnect_telemetry()',
                     'a.close()', 'Car.close_all()', 'gps.close()', 'print(gps.latest)',
                     'print(gps.port)', 'print(a.connection_generation)',
                     'input.INVERT_Y_AXIS=True', 'from vsss import GPS']
        for statement in forbidden:
            with self.subTest(statement=statement),self.assertRaises(ValueError):
                validate(program('a=Car(1)\n'+statement))
        # Bypass the HTTP validator deliberately: the RPC surface must also deny
        # configuration and unnormalized axes.
        for statement in ['a.set_trim()', 'a.set_controller({})', 'a.restart()', 'input.get_axis("LX0",normalize=False)']:
            with self.subTest(statement=statement):
                result=self.finish(self.runner.start(program('a=Car(1)\n'+statement))['run_id'])
                self.assertEqual(result['state'],'failed',result)
        result=self.finish(self.start('a=Car(1)\nb=Car(1)'))
        self.assertIn('already has a connection',result['error'])

    def test_camera_wait_output_limit_and_legacy_context(self):
        def factory(number,**kwargs):
            car=self.devices.car(number,**kwargs); car.pose=None; return car
        self.runner=LiveRunner(factory,cleanup=lambda:None)
        run=self.start('a=Car(1)\na.wait_for_position(timeout=60)')
        self.wait_for(lambda:len(self.devices.cars)==1)
        self.runner.stop(run)
        self.assertEqual(self.finish(run)['state'],'stopped')
        self.assert_closed()
        self.runner=self.devices.runner()
        result=self.finish(self.start('a=Car(1)\nprint("x" * 20000)'))
        self.assertEqual(result['state'],'finished',result)
        self.assertLessEqual(len(result['output']),16000)
        legacy='from car import Car\nwith Car(1) as car:\n    car.set_velocity(.2)\n'
        validate(legacy)
        self.assertEqual(self.finish(self.runner.start(legacy)['run_id'])['state'],'finished')

    def test_python_basics_examples(self):
        expected={'01_print_and_add.py':'Hello, Python!\n12\n',
                  '02_variables_and_loops.py':'2\n4\n6\n8\n10\n'}
        for name,output in expected.items():
            code=(ROOT/'blockly_python_demo/examples'/name).read_text()
            validate(code)
            result=self.finish(self.runner.start(code)['run_id'])
            self.assertEqual(result['state'],'finished',result)
            self.assertEqual(result['output'],output)
        self.assertFalse(self.devices.cars)
        self.assertEqual(self.devices.field_reads,0)

    def test_partial_teleop_maps_keys_and_rx_independently(self):
        code=(ROOT/'blockly_python_demo/examples/06_partial_teleop.py').read_text()
        validate(code)
        self.devices.input.pressed_keys.clear()
        self.devices.input._axis_states[0]['RX']=0
        run=self.runner.start(code)['run_id']
        self.wait_for(lambda:self.devices.cars)
        for keys,rx,forward,turn in [({'w'},32767,.5,.5),({'s'},-32768,-.5,-.5),
                                     (set(),0,0,0),({'w','s'},32767,0,.5),({'w'},0,.5,0)]:
            with self.subTest(keys=keys,rx=rx):
                car=self.devices.cars[0]
                start=len(car.events)
                self.devices.input.pressed_keys.clear()
                self.devices.input.pressed_keys.update(keys)
                self.devices.input._axis_states[0]['RX']=rx
                self.wait_for(lambda:('velocity',forward,0,turn) in car.events[start:])
        self.runner.stop(run)
        self.assertEqual(self.finish(run)['state'],'stopped')
        self.assert_closed()
        self.assertEqual(self.devices.field_reads,0)

    def test_gamepad_reader_prints_buttons_sticks_and_triggers_without_car(self):
        self.devices.input._axis_states[0].update(RX=-32768,RY=32767,LT=0,RT=255)
        code=(ROOT/'blockly_python_demo/examples/07_read_gamepad.py').read_text()
        validate(code)
        run=self.runner.start(code)['run_id']
        self.wait_for(lambda:'Triggers LT/RT: 0.0 1.0' in self.runner.status(run)['output'])
        output=self.runner.status(run)['output']
        self.assertIn('Buttons A/B/X/Y: True False False False',output)
        self.assertIn('Sticks LX/LY/RX/RY: 0.3 0.5 -1.0 1.0',output)
        self.assertIn('Triggers LT/RT: 0.0 1.0',output)
        self.runner.stop(run)
        self.assertEqual(self.finish(run)['state'],'stopped')
        self.assertFalse(self.devices.cars)
        self.assertEqual(self.devices.field_reads,0)

    def test_infinite_rotation_alternates_until_stopped(self):
        self.runner=self.devices.runner()
        self.assertIsNone(self.runner.max_seconds)
        code=(ROOT/'blockly_python_demo/examples/03_rotate_back_and_forth.py').read_text()
        validate(code)
        run=self.runner.start(code)['run_id']
        self.wait_for(lambda:self.devices.cars and len([e for e in self.devices.cars[0].events if e[0]=='velocity'])>=3)
        events=[e for e in self.devices.cars[0].events if e[0]=='velocity']
        self.assertEqual(events[:3],[('velocity',0,0,.2),('velocity',0,0,-.2),('velocity',0,0,.2)])
        self.assertEqual(self.runner.status(run)['state'],'running')
        self.runner.stop(run)
        self.assertEqual(self.finish(run)['state'],'stopped')
        self.assertEqual(self.devices.field_reads,0)
        self.assert_closed()

    def test_position_example_only_reads_field(self):
        code=(ROOT/'blockly_python_demo/examples/04_read_car_position.py').read_text()
        validate(code)
        run=self.runner.start(code)['run_id']
        self.wait_for(lambda:'Pose(' in self.runner.status(run)['output'])
        self.runner.stop(run)
        self.assertEqual(self.finish(run)['state'],'stopped')
        self.assertFalse(self.devices.cars)
        self.assertEqual(self.devices.field_reads,1)

    def test_alternating_waypoints_wait_and_can_be_stopped(self):
        code=(ROOT/'blockly_python_demo/examples/05_alternate_waypoints.py').read_text()
        validate(code)
        run=self.runner.start(code)['run_id']
        self.wait_for(lambda:self.devices.cars and len([e for e in self.devices.cars[0].events if e[0]=='waypoint'])>=3)
        events=[e for e in self.devices.cars[0].events if e[0]=='waypoint']
        self.assertEqual(events[:3],[('waypoint',0,.5,90),('waypoint',0,-.5,90),('waypoint',0,.5,90)])
        self.runner.stop(run)
        self.assertEqual(self.finish(run)['state'],'stopped')
        self.assert_closed()

    def test_waypoint_fault_or_timeout_does_not_send_the_next_target(self):
        from vsss._ble_protocol import RobotStatus
        code=(ROOT/'blockly_python_demo/examples/05_alternate_waypoints.py').read_text()
        for fault,expected in ((1,'Onboard position/control unavailable'),(0,'Waypoint timed out')):
            with self.subTest(fault=fault):
                self.devices.cars.clear()
                def factory(number,**kwargs):
                    car=self.devices.car(number,**kwargs)
                    car.read_status=lambda:RobotStatus(2,3,fault,0,.1,.2,90.,10)
                    return car
                self.runner=LiveRunner(factory,cleanup=self.devices.cleanup)
                test_code=code.replace('monotonic() + 10','monotonic() + 0.15')
                result=self.finish(self.runner.start(test_code)['run_id'])
                self.assertEqual(result['state'],'failed',result)
                self.assertIn(expected,result['error'])
                self.assertEqual(len([e for e in self.devices.cars[0].events if e[0]=='waypoint']),1)
                self.assert_closed()



class SchemaTests(unittest.TestCase):
    def test_discovery_uses_current_signatures_and_never_imports_input(self):
        data=schema()
        axis=next(b for b in data['blocks'] if b['id']=='api_input_get_axis')
        self.assertEqual([(a['name'],a['default']) for a in axis['args']],[('axis_name',None)])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'car.py'
            path.write_text(SOURCES['car'].read_text(encoding='utf-8-sig').replace('def set_velocity(self, x=0., y=0., w=0.):','def set_velocity(self, x=0., y=0., w=0., gain=0.7):'),encoding='utf-8')
            with patch.dict(SOURCES,{'car':path}):
                velocity=next(b for b in schema()['blocks'] if b['id']=='api_car_set_velocity')
                self.assertEqual(velocity['args'][-1]['name'],'gain')
                self.assertEqual(velocity['args'][-1]['default'],.7)
        import sys
        self.assertNotIn('vsss.input',sys.modules)

    def test_validation_limits_exposure(self):
        for code in ('import os','from vsss import _provider','print(input._axis_states)','car._client.stop()',
                     'car.unknown()','gps.add_listener(print)','input.pressed_keys.clear()','car.pose=1'):
            with self.subTest(code=code),self.assertRaises(ValueError): validate(code)


if __name__=='__main__': unittest.main()
