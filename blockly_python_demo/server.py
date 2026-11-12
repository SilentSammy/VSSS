"""Local Blockly editor controlling real VSSS robots through the Car API."""
import argparse
import ast
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from live_runtime import LiveRunner

ROOT = Path(__file__).resolve().parent
CALLS = {'print', 'range', 'str', 'int', 'float', 'len', 'abs', 'round', 'min', 'max', 'sum', 'Car', 'sleep'}
METHODS = {'set_velocity', 'go_to', 'stop', 'wait_for_position', 'read_status'}
ATTRIBUTES = METHODS | {'pose', 'position', 'is_connected'}
NODES = (ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.Name, ast.Load,
         ast.Store, ast.Constant, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
         ast.If, ast.For, ast.While, ast.Break, ast.Continue, ast.Pass, ast.Call,
         ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.USub,
         ast.UAdd, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
         ast.Gt, ast.GtE, ast.List, ast.Tuple, ast.Subscript, ast.Slice,
         ast.ImportFrom, ast.alias, ast.With, ast.withitem, ast.Attribute,
         ast.keyword, ast.Is, ast.IsNot)


def validate(code):
    if not isinstance(code, str) or len(code) > 20000:
        raise ValueError('Program must be text of at most 20,000 characters.')
    tree = ast.parse(code)
    sessions = [node for node in ast.walk(tree) if isinstance(node, ast.With)]
    if len(sessions) > 1:
        raise ValueError('Use one car session per program.')
    constructors = {id(item.context_expr) for node in sessions for item in node.items}
    for node in ast.walk(tree):
        if not isinstance(node, NODES):
            raise ValueError(f'{type(node).__name__} is outside this demo’s supported Python subset.')
        if isinstance(node, ast.ImportFrom) and not (
                node.level == 0 and (node.module, tuple((a.name, a.asname) for a in node.names)) in
                {('car', (('Car', None),)), ('time', (('sleep', None),))}):
            raise ValueError('Only from car import Car and from time import sleep are supported.')
        if isinstance(node, ast.Attribute) and not (
                isinstance(node.value, ast.Name) and node.value.id == 'car'
                and node.attr in ATTRIBUTES and isinstance(node.ctx, ast.Load)):
            raise ValueError('Only the student-facing car methods and pose properties are supported.')
        if isinstance(node, ast.With):
            if len(node.items) != 1:
                raise ValueError('Use one car per session.')
            item = node.items[0]
            if not (isinstance(item.context_expr, ast.Call)
                    and isinstance(item.context_expr.func, ast.Name)
                    and item.context_expr.func.id == 'Car'
                    and isinstance(item.optional_vars, ast.Name) and item.optional_vars.id == 'car'):
                raise ValueError('Use with Car(number) as car.')
        if isinstance(node, ast.Name) and (node.id.startswith('_') or
                isinstance(node.ctx, ast.Store) and node.id in CALLS):
            raise ValueError('That variable name is reserved.')
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == 'Car' and id(node) not in constructors:
                raise ValueError('Create a car only with: with Car(number) as car.')
            basic = isinstance(node.func, ast.Name) and node.func.id in CALLS
            method = (isinstance(node.func, ast.Attribute) and node.func.attr in METHODS
                      and isinstance(node.func.value, ast.Name) and node.func.value.id == 'car')
            if not (basic or method):
                raise ValueError('Only basic built-ins and student-facing car methods may be called.')


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        # Expose only the demo page and Blockly's browser assets.
        path = self.path.split('?', 1)[0]
        allowed = {'/', '/index.html', '/app.js', '/vsss_blocks.js', '/style.css'}
        if path not in allowed and not path.startswith('/node_modules/blockly/'):
            self.send_error(404)
            return
        if '..' in path or '%' in path:
            self.send_error(404)
            return
        super().do_GET()

    def do_POST(self):
        if self.path not in {'/run', '/status', '/stop'}:
            self.send_error(404)
            return
        # Requests must originate in this editor, not an unrelated web page.
        host = self.headers.get('Host', '')
        expected = f'http://{host}'
        valid_hosts = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        if host not in valid_hosts or self.headers.get('Origin') != expected or self.headers.get('X-Blockly-Demo') != '1':
            self.send_error(403)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 40000:
                raise ValueError('Request is too large or empty.')
            payload = json.loads(self.rfile.read(length))
            if self.path == '/run':
                validate(payload.get('code'))
                result = self.server.runner.start(payload['code'])
            elif self.path == '/stop':
                result = self.server.runner.stop(payload.get('run_id'))
            else:
                result = self.server.runner.status(payload.get('run_id'))
        except (ValueError, SyntaxError, TypeError, AttributeError) as exc:
            self.reply({'error': str(exc)}, status=400)
            return
        self.reply(result)

    def reply(self, result, status=200):
        data = json.dumps(result).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)


def serve(port=8765, runner=None):
    runner = runner or LiveRunner()
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.runner = runner
    print(f'VSSS Blocks LIVE: http://127.0.0.1:{server.server_port}', flush=True)
    print('Run controls real hardware. Press Ctrl+C to stop and disconnect.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        runner.shutdown()
        server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    serve(args.port)
