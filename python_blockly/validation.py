"""Validate the small generated Python language and its approved API surface."""
import ast
from api_registry import CAR_METHODS, CAR_PROPERTIES, GPS_METHODS, GPS_PROPERTIES, INPUT_METHODS, INPUT_PROPERTIES, RECORD_FIELDS

BUILTINS = {'print', 'range', 'str', 'int', 'float', 'bool', 'len', 'abs', 'round', 'min', 'max',
            'sum', 'dict', 'list', 'tuple', 'enumerate', 'sorted', 'any', 'all',
            'RuntimeError', 'TimeoutError', 'ValueError'}
CALLS = BUILTINS | {'Car', 'sleep', 'monotonic', 'main'}
METHODS = set(CAR_METHODS) | GPS_METHODS | INPUT_METHODS | {'get', 'keys', 'values', 'items', 'append'}
ATTRIBUTES = METHODS | CAR_PROPERTIES | GPS_PROPERTIES | INPUT_PROPERTIES | RECORD_FIELDS
NODES = (ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.Name, ast.Load, ast.Store,
         ast.Constant, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.If, ast.For,
         ast.While, ast.Break, ast.Continue, ast.Pass, ast.Call, ast.Add, ast.Sub, ast.Mult,
         ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
         ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn,
         ast.List, ast.Tuple, ast.Dict, ast.Subscript, ast.Slice, ast.ImportFrom, ast.alias,
         ast.With, ast.withitem, ast.Attribute, ast.keyword, ast.Starred, ast.FunctionDef,
         ast.arguments, ast.Return, ast.Raise, ast.JoinedStr, ast.FormattedValue)


def validate(code):
    if not isinstance(code, str) or len(code) > 40000:
        raise ValueError('Program must be text of at most 40,000 characters.')
    tree = ast.parse(code)
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if len(functions) > 1:
        raise ValueError('Use one @run main program.')
    for node in ast.walk(tree):
        if not isinstance(node, NODES):
            raise ValueError(f'{type(node).__name__} is outside the supported block language.')
        if isinstance(node, ast.FunctionDef):
            if (node not in tree.body or node.name != 'main' or node.args.args or node.args.posonlyargs
                    or node.args.kwonlyargs or node.args.vararg or node.args.kwarg
                    or len(node.decorator_list) != 1 or not isinstance(node.decorator_list[0], ast.Name)
                    or node.decorator_list[0].id != 'run'):
                raise ValueError('Only a synchronous @run def main() entry point is supported.')
        if isinstance(node, ast.ImportFrom):
            approved = {'vsss': {'Car', 'gps', 'run', 'input'},
                        'time': {'sleep', 'monotonic'}, 'car': {'Car'}}
            if node.level or node.module not in approved or any(
                    a.asname or a.name not in approved[node.module] for a in node.names):
                raise ValueError('Import only the approved vsss API and time helpers.')
        if isinstance(node, ast.Attribute):
            if node.attr not in ATTRIBUTES or not isinstance(node.ctx, ast.Load):
                raise ValueError('Only approved public API methods and data fields are supported.')
        if isinstance(node, ast.Name):
            if node.id.startswith('_') and not (node.id == '__name__' and isinstance(node.ctx, ast.Load)):
                raise ValueError('Private names are not available to block programs.')
            if isinstance(node.ctx, ast.Store) and node.id in CALLS | {'run', 'gps', 'input'}:
                raise ValueError(f'{node.id} is reserved.')
        if isinstance(node, ast.With):
            for item in node.items:
                if not (isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Name)
                        and item.context_expr.func.id == 'Car' and isinstance(item.optional_vars, ast.Name)):
                    raise ValueError('Only Car context managers are supported.')
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in CALLS:
                    raise ValueError('Only approved functions may be called.')
            elif not isinstance(node.func, ast.Attribute) or node.func.attr not in METHODS:
                raise ValueError('Only approved methods may be called.')
    if functions:
        main_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Name) and n.func.id == 'main']
        if len(main_calls) != 1 or main_calls[0].args or main_calls[0].keywords:
            raise ValueError('Call main() exactly once.')
    return tree
