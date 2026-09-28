"""CPU-only ownership, failure and handoff regression tests."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import train_then_burn as handoff


class HandoffTests(unittest.TestCase):
    def test_progress_requires_all_ranks_and_advancing_collectives(self):
        ready = '\n'.join(f'gpu_burn_ready rank={r} device=NVIDIA B200 world_size=8 collective_probe_sum=36.0 comm_total_mib=1137' for r in range(8))
        p1 = '\ngpu_burn_progress rank=0 completed_cycles=10 completed_gemms=300 completed_collective_payload_gib=11.10 average_cycle_seconds=0.8'
        p2 = '\ngpu_burn_progress rank=0 completed_cycles=20 completed_gemms=600 completed_collective_payload_gib=22.21 average_cycle_seconds=0.8'
        self.assertFalse(handoff.burn_progress(ready + p1))
        self.assertFalse(handoff.burn_progress(ready + p1 + p1))
        self.assertFalse(handoff.burn_progress(ready.replace('rank=7', 'rank=6') + p1 + p2))
        self.assertTrue(handoff.burn_progress(ready + p1 + p2))

    def test_real_queue_success_and_failure_both_handoff(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                job = {'name': 'test', 'argv': ['{python}', '-c',
                    'raise SystemExit(7)' if fail else "from pathlib import Path; Path('" + str(root / 'run/result.json') + "').write_text('{}')"],
                    'required_outputs': [{'path': 'result.json'}]}
                with patch.object(handoff, 'clean_owned_children') as cleanup, patch.object(handoff.time, 'sleep'), \
                     patch.object(handoff, 'require_free', return_value=[]), \
                     patch.object(handoff, 'start_burn', return_value={'verified': True}) as burn, \
                     patch.object(handoff.signal, 'signal'):
                    code = handoff.execute_queue([job], Path.cwd(), root, 'test')
                cleanup.assert_called_once()
                burn.assert_called_once()
                self.assertEqual(code, 1 if fail else 0)
                receipt = json.loads((root / 'supervisor.json').read_text())
                self.assertEqual(receipt['training_status'], 'failed' if fail else 'ok')
                self.assertEqual((root / 'run/complete.json').exists(), not fail)

    def test_busy_gpu_prevents_burn(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(handoff, 'run_jobs', return_value=0), \
             patch.object(handoff, 'clean_owned_children'), patch.object(handoff.time, 'sleep'), \
             patch.object(handoff, 'require_free', side_effect=RuntimeError('unknown GPU worker')), \
             patch.object(handoff, 'start_burn') as burn, patch.object(handoff.signal, 'signal'):
            self.assertEqual(handoff.execute_queue([], Path.cwd(), Path(tmp), 'test'), 1)
            burn.assert_not_called()

    def test_unknown_gpu_worker_never_signaled(self):
        gpu = [{'index': 0, 'uuid': 'test', 'pids': [123]}]
        with tempfile.TemporaryDirectory() as tmp, patch.object(handoff, 'snapshot', return_value=gpu), \
             patch.object(handoff, 'process', return_value={'ppid': 124}), \
             patch.object(handoff, 'approved_launcher', return_value=False), \
             patch.object(handoff, 'pidfd_send_signal') as send:
            with self.assertRaisesRegex(RuntimeError, 'not an approved burn'):
                handoff.stop_burns(Path(tmp), {})
            send.assert_not_called()

    def test_subreaper_cleans_owned_orphans_only(self):
        # Isolate subreaper and signal side effects in a fresh Python process.
        # An unrelated sibling sleeper must survive the owned orphan cleanup.
        other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        try:
            subprocess.run([sys.executable, '-c', '''
import os, subprocess, sys, time
from scripts.train_then_burn import enable_subreaper, descendants, clean_owned_children
enable_subreaper()
p = subprocess.Popen([sys.executable, '-c', "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)"])
p.wait()
assert descendants(), 'orphan not adopted'
clean_owned_children()
assert not descendants(), 'owned children remain'
'''], check=True, timeout=40)
            self.assertIsNone(other.poll())
        finally:
            other.terminate()
            other.wait(timeout=10)


if __name__ == '__main__':
    unittest.main()
