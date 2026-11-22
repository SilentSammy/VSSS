"""Explicit student API approval and presentation; signatures come from source.

Reading AST rather than importing modules keeps discovery free of input listeners,
provider connections, Bluetooth and GPS side effects.
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = {name: ROOT / 'vsss' / f'{name}.py' for name in ('car', 'gps', 'input')}
# Student approval only. Instructor commands remain in the normal Python/CLI API.
CAR_METHODS = {
    'set_velocity': ('Movement', 'set velocity', False),
    'go_to': ('Movement', 'go to field waypoint', False),
    'stop': ('Movement', 'stop', False),
    'wait_for_position': ('Advanced lessons', 'wait for camera position', False),
    'read_status': ('Advanced lessons', 'robot status', True),  # Used by readable status blocks, not shown raw.
}
CAR_PROPERTIES = {'number', 'pose', 'position'}
GPS_PROPERTIES = {'cars', 'ball', 'balls', 'board_detected', 'feedback_problem'}
GPS_METHODS = {'get_car'}
INPUT_METHODS = {'is_pressed', 'is_toggled', 'rising_edge', 'falling_edge', 'get_axis', 'get_bipolar_ctrl'}
INPUT_PROPERTIES = set()
RECORD_FIELDS = {'x', 'y', 'angle', 'mode_name', 'fault_name', 'pose_fresh'}
DERIVED_FIELDS = ('mode_name', 'fault_name', 'pose_fresh')


def signatures(owner):
    tree = ast.parse(SOURCES[owner].read_text(encoding='utf-8-sig'))
    nodes = tree.body
    if owner != 'input':
        class_name = 'GPS' if owner == 'gps' else 'Car'
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    return {n.name: n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def parameters(node):
    positional = node.args.posonlyargs + node.args.args
    defaults = [None] * (len(positional) - len(node.args.defaults)) + node.args.defaults
    pairs = list(zip(positional, defaults)) + list(zip(node.args.kwonlyargs, node.args.kw_defaults))
    result = []
    for arg, default_node in pairs:
        if arg.arg in {'self', 'cls'}:
            continue
        required = default_node is None
        default = None if required else ast.literal_eval(default_node)
        annotation = ast.unparse(arg.annotation) if arg.annotation else ''
        name = arg.arg
        if annotation == 'bool' or isinstance(default, bool):
            kind = 'Boolean'
        elif annotation == 'str' or isinstance(default, str) or name in {'input_name', 'axis_name', 'high_key', 'low_key', 'high_game', 'low_game'}:
            kind = 'String'
        elif annotation in {'int', 'float'} or type(default) in (int, float) or name in {'x', 'y', 'w', 'angle', 'timeout', 'number'}:
            kind = 'Number'
        else:
            kind = None
        example = default
        if required:
            example = {'input_name': 'w', 'axis_name': 'LX0'}.get(name, 0)
        result.append(dict(name=name, type=kind, default=default, required=required, example=example))
    return result


def schema():
    blocks = []
    all_nodes = {owner: signatures(owner) for owner in SOURCES}
    for name, (category, label, returns) in CAR_METHODS.items():
        if name == 'read_status':
            continue  # The UI uses named reached / pose / fault checks instead.
        node = all_nodes['car'][name]  # Removed/renamed APIs fail discovery visibly.
        blocks.append(dict(id=f'api_car_{name}', owner='car', member=name, kind='method',
                           category=category, label=label, returns=returns,
                           args=parameters(node), tooltip=ast.get_docstring(node) or label))
    for owner, names in (('car', CAR_PROPERTIES), ('gps', GPS_PROPERTIES), ('input', INPUT_PROPERTIES)):
        for name in sorted(names):
            node = all_nodes[owner].get(name)
            blocks.append(dict(id=f'api_{owner}_{name}', owner=owner, member=name, kind='property',
                               category='Car feedback' if owner == 'car' else ('Field' if owner == 'gps' else 'Advanced lessons'),
                               label='ball position' if owner == 'gps' and name == 'ball' else name.replace('_', ' '), returns=True, args=[],
                               tooltip=(ast.get_docstring(node) if node else None) or name))
    for owner, names in (('gps', GPS_METHODS), ('input', INPUT_METHODS)):
        for name in sorted(names):
            node = all_nodes[owner][name]
            blocks.append(dict(id=f'api_{owner}_{name}', owner=owner, member=name, kind='method',
                               category='Field' if owner == 'gps' else 'Advanced lessons', label=name.replace('_', ' '),
                               returns=True, args=[a for a in parameters(node) if not (owner == 'input' and name == 'get_axis' and a['name'] == 'normalize')],
                               tooltip=ast.get_docstring(node) or name))
    return dict(version=3, blocks=blocks, record_fields=['x', 'y', 'angle'],
                constructor=[p for p in parameters(all_nodes['car']['__init__']) if p['name'] in {'number', 'timeout'}])


if __name__ == '__main__':
    print(json.dumps(schema(), indent=2))
