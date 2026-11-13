"""Test-only I/O fixtures. No motion model, listeners, provider, or BLE imports."""
import ast
from pathlib import Path
from types import MappingProxyType

from live_runtime import LiveRunner
from vsss.gps import Pose, Position, Feedback
from vsss._ble_protocol import RobotStatus


class FakeCar:
    def __init__(self, number, *, timeout=15):
        self.number = number
        self.pose = Pose(.1 * number, .2, 90.)
        self.is_connected = True
        self.connection_generation = 1
        self.events = [('connect', number)]
        self.trim_matrix = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        self.controller_config = {'horizon': .15}

    @property
    def position(self):
        return None if self.pose is None else (self.pose.x, self.pose.y)

    def set_velocity(self, x=0., y=0., w=0.):
        if not all(-1 <= v <= 1 for v in (x, y, w)):
            raise ValueError('velocity must be within [-1, 1]')
        self.events.append(('velocity', x, y, w))

    def go_to(self, x, y, angle):
        self.events.append(('waypoint', x, y, angle))
        return x, y, angle

    def read_status(self):
        self.events.append(('status',))
        return RobotStatus(3, 3, 0, 0, self.number * .1, .2, 90., 10)

    def stop(self):
        self.events.append(('stop',))

    def close(self):
        self.events.append(('close',))
        self.is_connected = False

    def set_trim(self, matrix=None):
        self.trim_matrix = matrix or ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        return self.trim_matrix

    def read_trim(self):
        return self.trim_matrix

    def set_controller(self, parameters):
        self.controller_config.update(parameters)
        return self.controller_config

    def read_controller(self):
        return self.controller_config

    def restart(self, timeout=45):
        self.events.append(('restart',))
        self.connection_generation += 1

    def reconnect_telemetry(self):
        self.events.append(('reconnect_telemetry',))


class FakeGPS:
    cars = {1: Pose(.1, .2, 90.), 2: Pose(.2, .2, 90.)}
    balls = (Position(.3, .4),)
    board_detected = True
    feedback_problem = None
    port = configured_port = 5000
    stale_after = .5
    subscribe = True
    latest = Feedback(1., True, MappingProxyType(cars), balls)

    def __init__(self):
        self.closes = 0

    def get_car(self, number):
        return self.cars.get(number)

    def close(self):
        self.closes += 1


def native_input_functions():
    """Use the actual API implementations, without module-level device threads."""
    source = Path(__file__).resolve().parent.parent / 'vsss/input.py'
    tree = ast.parse(source.read_text(encoding='utf-8-sig'))
    functions = {'_parse_input', 'is_pressed', 'is_toggled', 'rising_edge', 'falling_edge', 'get_axis', 'get_bipolar_ctrl'}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in functions]
    names = {'A', 'B', 'X', 'Y', 'LX', 'LY', 'RX', 'RY', 'LT', 'RT', 'L3', 'R3', 'DPAD_X', 'DPAD_Y'}
    namespace = dict(_ALL_KNOWN_NAMES=names, _ABS_TO_NAME={'ABS_X':'LX', 'ABS_Y':'LY', 'ABS_RX':'RX', 'ABS_RY':'RY'},
                     _pressed_buttons={0:{'A'}, 1:{'B'}}, _just_pressed_buttons={0:{'A'}, 1:{'B'}},
                     _just_released_buttons={0:set(), 1:{'X'}}, _toggles={0:{'A':True}, 1:{}},
                     _axis_states={0:{'LX':8192,'LY':16384},1:{'LX':-8192,'LY':-16384}},
                     pressed_keys={'w'}, just_pressed_keys={'w'}, just_released_keys={'s'}, toggles={}, INVERT_Y_AXIS=False,
                     KEYBOARD_AVAILABLE=True)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)

    class API:
        def __getattr__(self, name):
            return namespace[name]

        def __setattr__(self, name, value):
            namespace[name] = value

    return API()


class Devices:
    def __init__(self):
        self.cars = []
        self.gps = FakeGPS()
        self.input = native_input_functions()
        self.field_reads = self.input_reads = 0

    def car(self, number, **kwargs):
        car = FakeCar(number, **kwargs)
        self.cars.append(car)
        return car

    def field(self):
        self.field_reads += 1
        return self.gps

    def inputs(self):
        self.input_reads += 1
        return self.input

    def cleanup(self):
        if self.field_reads:
            self.gps.close()

    def runner(self, **kwargs):
        return LiveRunner(self.car, gps_factory=self.field, input_factory=self.inputs,
                          cleanup=self.cleanup, **kwargs)
