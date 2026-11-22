"""Installed command, browser launch, and package import checks."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.request import Request, urlopen

from python_blockly import server


class CliTest(unittest.TestCase):
    def test_main_opens_browser_unless_disabled(self):
        with patch.object(server, 'serve') as serve:
            server.main([])
            self.assertTrue(serve.call_args.kwargs['open_browser'])
            server.main(['--no-browser', '--port', '8767'])
            self.assertFalse(serve.call_args.kwargs['open_browser'])
            self.assertEqual(serve.call_args.args[0], 8767)

    def test_browser_receives_bound_server_url(self):
        opened = Event()
        http = Mock(server_port=32123)
        def run():
            self.assertTrue(opened.wait(2))
            raise KeyboardInterrupt()
        http.serve_forever.side_effect = run
        with patch.object(server, 'ThreadingHTTPServer', return_value=http), \
             patch('webbrowser.open', side_effect=lambda url: opened.set() or True) as browser:
            server.serve(port=0, runner=SimpleNamespace(shutdown=lambda: None), open_browser=True)
        browser.assert_called_once_with('http://127.0.0.1:32123')
        http.server_close.assert_called_once()

    def test_installed_command_serves_editor_from_another_directory(self):
        executable = Path(sys.executable).with_name('vsss-blockly.exe' if sys.platform == 'win32' else 'vsss-blockly')
        with tempfile.TemporaryDirectory() as directory:
            process = subprocess.Popen([str(executable), '--no-browser', '--port', '0',
                '--endpoint', str(Path(directory)/'rpc.json'), '--provider-port', '0'],
                cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                line = process.stdout.readline()
                self.assertIn('http://127.0.0.1:', line)
                url = line.strip().split()[-1]
                with urlopen(url, timeout=4) as response:
                    html=response.read().decode()
                    self.assertIn('Program your car with blocks.', html)
                    self.assertRegex(html,r'vsss_blocks\.js\?v=[0-9a-f]{12}')
                    self.assertRegex(html,r'app\.js\?v=[0-9a-f]{12}')
                for resource in ('/app.js', '/node_modules/blockly/blockly_compressed.js', '/api/schema'):
                    with urlopen(url+resource, timeout=4) as response:
                        self.assertEqual(response.status, 200)
                request = Request(url+'/validate', data=json.dumps({'code': 'print(42)'}).encode(), headers={
                    'Content-Type': 'application/json', 'Origin': url, 'X-VSSS-Blockly': '1'})
                with urlopen(request, timeout=4) as response:
                    self.assertTrue(json.load(response)['valid'])
                self.assertFalse((Path(directory)/'rpc.json').exists())
                headers = {'Content-Type': 'application/json', 'Origin': url, 'X-VSSS-Blockly': '1'}
                request = Request(url+'/run', data=json.dumps({'code': 'print(42)'}).encode(), headers=headers)
                with urlopen(request, timeout=5) as response: started = json.load(response)
                deadline = time.monotonic()+5
                while True:
                    request = Request(url+'/status', data=json.dumps({'run_id': started['run_id']}).encode(), headers=headers)
                    with urlopen(request, timeout=4) as response: result = json.load(response)
                    if result['state'] not in ('running', 'stopping'): break
                    self.assertLess(time.monotonic(), deadline, result)
                    time.sleep(.05)
                self.assertEqual(result['state'], 'finished', result)
                self.assertEqual(result['output'], '42\n')
            finally:
                process.terminate()
                process.communicate(timeout=5)
