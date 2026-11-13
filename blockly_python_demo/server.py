"""Local Blockly editor controlling real VSSS robots through the Car API."""
import argparse
import ast
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from live_runtime import LiveRunner

ROOT = Path(__file__).resolve().parent
from validation import validate
from api_registry import schema


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        if self.path == '/api/schema':
            try:
                self.reply(schema())
            except (KeyError, ValueError, StopIteration) as exc:
                self.reply({'error': f'API registry no longer matches VSSS: {exc}'}, status=500)
            return
        # Expose only the editor and Blockly browser assets.
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
        if self.path not in {'/run', '/status', '/stop', '/validate'}:
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
            if not 0 < length <= 160000:
                raise ValueError('Request is too large or empty.')
            payload = json.loads(self.rfile.read(length))
            if self.path == '/validate':
                validate(payload.get('code'))
                result = {'valid': True}
            elif self.path == '/run':
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
