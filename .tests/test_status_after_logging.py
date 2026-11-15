"""Exercise the firmware loop's actual logging/status tail without hardware."""
import ast
from pathlib import Path
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class StatusAfterLoggingTests(unittest.TestCase):
    def run_tail(self, *, queued=True, flush=True):
        tree = ast.parse(Path('mp/main.py').read_text())
        # Execute the production statements, avoiding hardware initialization.
        loop = next(node for node in ast.walk(tree) if isinstance(node, ast.While)
                    and any(isinstance(n, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == 'previous_loop'
                        for t in n.targets) for n in node.body))
        start = next(i for i, node in enumerate(loop.body)
                     if isinstance(node, ast.If) and isinstance(node.test, ast.Call)
                     and isinstance(node.test.func, ast.Attribute)
                     and node.test.func.attr == 'sample')
        code = compile(ast.Module(body=loop.body[start:start + 2], type_ignores=[]),
                       'main-status-tail', 'exec')
        clock = [1000]
        class Receiver:
            pose = (0., .4, 90.)
            packets = 22
            received = 1000
            @property
            def fresh(self):
                return clock[0] - self.received + 105 <= 500
        receiver = Receiver()
        def sample(*args):
            if flush:
                clock[0] += 460
            return flush
        def poll():
            if queued:
                receiver.received = clock[0]
                receiver.packets += 4
        telemetry = SimpleNamespace(receiver=receiver, poll=Mock(side_effect=poll))
        ble = SimpleNamespace(set_value=Mock())
        env = dict(nav_log=SimpleNamespace(sample=sample), telemetry=telemetry,
                   receiver=receiver, time=SimpleNamespace(ticks_ms=lambda: clock[0],
                   ticks_diff=lambda a,b: a-b), now=1000, last_status=900,
                   state=dict(mode=3, fault=0), controller=None, car=None,
                   wlan=SimpleNamespace(isconnected=lambda: True), loop_ms=10,
                   ble_server=ble, STATUS_UUID='status', struct=struct)
        exec(code, env)
        return telemetry, struct.unpack('<BBBBfffI', ble.set_value.call_args.args[1])

    def test_queued_feedback_is_refreshed_after_460ms_flush(self):
        telemetry, status = self.run_tail()
        telemetry.poll.assert_called_once()
        self.assertEqual(status[1], 3)
        self.assertAlmostEqual(status[5], .4)
        self.assertEqual(status[-1], 26)

    def test_real_feedback_loss_is_not_masked(self):
        _, status = self.run_tail(queued=False)
        self.assertEqual(status[1], 1)
        self.assertEqual(status[4:7], (0., 0., 0.))

    def test_no_extra_poll_without_flash_work(self):
        telemetry, status = self.run_tail(flush=False)
        telemetry.poll.assert_not_called()
        self.assertEqual(status[1], 3)
