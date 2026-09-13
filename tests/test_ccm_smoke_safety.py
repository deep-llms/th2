"""Mock-only tests: never inspect or signal real GPUs/processes."""
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from scripts import gpu_status

with patch.dict(sys.modules, {'gpu_status': gpu_status}):
    spec = importlib.util.spec_from_file_location('smoke_gpu_control', Path(__file__).parents[1]/'scripts/ccm_smoke_gpu_control.py')
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)


class SmokeSafety(unittest.TestCase):
    def run_control(self, stale=False, extra=False, action='stop-verified-original'):
        status = [dict(name='NVIDIA B200', pids=[496+i], utilization_percent=98) for i in range(8)]
        if extra:
            status[0]['pids'].append(999)
        def ident(pid):
            if pid == 429:
                return 1, '179777866', [b'/usr/bin/python3', b'/tmp/llm_pretrain_burn.py']
            return 429, 'changed' if stale else '179777986', [b'--multiprocessing-fork']
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', ['control', action]))
            stack.enter_context(patch.object(control.socket, 'gethostname', return_value='thiennh-p6-8mgy-worker-0'))
            stack.enter_context(patch.object(control, 'snapshot', return_value=status))
            stack.enter_context(patch.object(control, 'identity', side_effect=ident))
            stack.enter_context(patch.object(control.Path, 'read_bytes', return_value=b'mocked script'))
            stack.enter_context(patch.object(control.Path, 'is_file', return_value=True))
            digest = stack.enter_context(patch.object(control.hashlib, 'sha256'))
            digest.return_value.hexdigest.return_value = '3cdcc857bd01b096e20a02640fa85f0b8be7607e3c2b22a89a704bbac3650857'
            opened = stack.enter_context(patch.object(control.os, 'pidfd_open', side_effect=lambda pid: pid+1000, create=True))
            sent = stack.enter_context(patch.object(control.signal, 'pidfd_send_signal', create=True))
            closed = stack.enter_context(patch.object(control.os, 'close'))
            if stale or extra:
                with self.assertRaises(RuntimeError):
                    control.main()
                sent.assert_not_called()
            else:
                control.main()
            return opened.call_args_list, sent.call_args_list, closed.call_args_list

    def test_only_verified_worker_handles_are_signaled(self):
        opened, sent, closed = self.run_control()
        self.assertEqual([c.args[0] for c in opened], list(range(496, 504)))
        self.assertEqual([c.args for c in sent], [(i+1000, control.signal.SIGKILL) for i in range(496, 504)])
        self.assertEqual([c.args[0] for c in closed], list(range(1496, 1504)))

    def test_stale_or_extra_process_refuses_all_signals(self):
        self.run_control(stale=True)
        self.run_control(extra=True)

    def test_burn_verification_is_read_only(self):
        opened, sent, closed = self.run_control(action='verify-burn')
        self.assertEqual((opened, sent, closed), ([], [], []))


if __name__ == '__main__':
    unittest.main()
