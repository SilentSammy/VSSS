"""Local Python Blockly editor with VSSS providers and an embedded field plot."""
import argparse
import hashlib
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
if __package__:
    from .live_runtime import LiveRunner
    from .validation import validate
    from .api_registry import schema
    from .environment import Environment
else:
    from live_runtime import LiveRunner
    from validation import validate
    from api_registry import schema
    from environment import Environment

ROOT = Path(__file__).resolve().parent
from threading import Lock


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        # Avoid mixing cached Blockly lesson definitions with an updated UI.
        if self.path.split('?', 1)[0] in {'/', '/index.html', '/app.js', '/vsss_blocks.js', '/style.css'}:
            self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def do_GET(self):
        try:
            self.get_response()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self.reply({'error': f'{type(exc).__name__}: {exc}'}, status=503)

    def get_response(self):
        if self.path.split('?',1)[0] in ('/', '/index.html'):
            # A fresh HTML page must fetch matching lesson/UI code even when a
            # browser cached older scripts before no-store was introduced.
            html = (ROOT/'index.html').read_text(encoding='utf-8')
            for name in ('app.js','vsss_blocks.js','style.css'):
                version = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()[:12]
                html = html.replace(f'"{name}"', f'"{name}?v={version}"')
            data = html.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type','text/html; charset=utf-8')
            self.send_header('Content-Length',str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path == '/api/environment':
            self.reply(self.server.environment.status() if self.server.environment else dict(options=[], mode='test', selected=None))
            return
        if self.path == '/api/plot':
            if self.server.environment is None:
                self.send_error(404)
                return
            import cv2
            with self.server.plot_lock:
                if self.server.plot is None:
                    from vsss.plot import EmbeddedPlot
                    self.server.plot = EmbeddedPlot(self.server.environment.endpoint)
                frame, metadata = self.server.plot.frame()
                ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                self.send_error(500)
                return
            data = encoded.tobytes()
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-VSSS-State', json.dumps(metadata))
            self.end_headers()
            self.wfile.write(data)
            return
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
        if self.path not in {'/run', '/status', '/stop', '/validate', '/environment', '/plot/pointer'}:
            self.send_error(404)
            return
        # Requests must originate in this editor, not an unrelated web page.
        host = self.headers.get('Host', '')
        expected = f'http://{host}'
        valid_hosts = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        if host not in valid_hosts or self.headers.get('Origin') != expected or self.headers.get('X-VSSS-Blockly') != '1':
            self.send_error(403)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 160000:
                raise ValueError('Request is too large or empty.')
            payload = json.loads(self.rfile.read(length))
            if self.path == '/environment':
                with self.server.action_lock:
                    if self.server.runner.status()['state'] in ('running', 'stopping'):
                        raise ValueError('Stop the program before changing the environment.')
                    if self.server.environment is None: raise ValueError('Environment selection unavailable')
                    result = self.server.environment.select(payload.get('selection'))
            elif self.path == '/plot/pointer':
                if self.server.plot is None: raise ValueError('Wait for the plot to load')
                self.server.plot.pointer(**payload)
                result = dict(ok=True)
            elif self.path == '/validate':
                validate(payload.get('code'))
                result = {'valid': True}
            elif self.path == '/run':
                validate(payload.get('code'))
                with self.server.action_lock:
                    if self.server.runner.status()['state'] in ('running', 'stopping'):
                        raise ValueError('A program is already running or stopping.')
                    if self.server.environment: self.server.environment.ensure()
                    result = self.server.runner.start(payload['code'])
            elif self.path == '/stop':
                result = self.server.runner.stop(payload.get('run_id'))
            else:
                result = self.server.runner.status(payload.get('run_id'))
        except Exception as exc:
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


def serve(port=8765, runner=None, *, endpoint=None, provider_port=38475, open_browser=False):
    testing = runner is not None
    if endpoint:
        import os
        os.environ['VSSS_ENDPOINT'] = str(endpoint)
    runner = runner or LiveRunner()
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.runner = runner
    server.environment = None if testing else Environment(endpoint, port=provider_port)
    server.plot, server.plot_lock, server.action_lock = None, Lock(), Lock()
    print(f'VSSS Python Blockly: http://127.0.0.1:{server.server_port}', flush=True)
    print('Choose Live or a simulator scenario. Press Ctrl+C to stop.', flush=True)
    browser_timer = None
    if open_browser:
        from threading import Timer
        import webbrowser
        def launch_browser():
            try:
                if not webbrowser.open(f'http://127.0.0.1:{server.server_port}'):
                    print('Open the URL above in your browser.', flush=True)
            except Exception as error:
                print(f'Could not open a browser: {error}. Open the URL above.', flush=True)
        browser_timer = Timer(.25, launch_browser)
        browser_timer.daemon = True
        browser_timer.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if browser_timer: browser_timer.cancel()
        try:
            runner.shutdown()
        finally:
            if server.plot: server.plot.close()
            if server.environment: server.environment.close()
            server.server_close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--endpoint', help='Provider discovery file; shared by editor, plot and student programs')
    parser.add_argument('--provider-port', type=int, default=38475, help='RPC port for providers started here; 0 selects a free port')
    parser.add_argument('--no-browser', action='store_true', help='Print the editor URL without opening a browser')
    args = parser.parse_args(argv)
    if not (ROOT / 'node_modules/blockly/blockly_compressed.js').is_file():
        parser.error(f'Blockly browser assets are missing. Run: npm.cmd ci --prefix "{ROOT}" --omit=dev')
    try:
        serve(args.port, endpoint=args.endpoint, provider_port=args.provider_port, open_browser=not args.no_browser)
    except OSError as error:
        parser.exit(1, f'Cannot start Blockly server: {error}. Try --port 8767 if the port is in use.\n')


if __name__ == '__main__':
    main()
