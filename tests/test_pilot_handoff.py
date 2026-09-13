"""CPU/mocked tests: never run GPU-management actions on the test machine."""
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import pilot_gpu_ops as gpu
import pilot_overnight as job
from validate_pilot_handoff import validate_log
from ccm.contracts import schedule


class Gates(unittest.TestCase):
    def test_training_log(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'log'
            records = [dict(step=i, input_tokens=i*16, nll=2., grad_norm=1.,
                            lr=schedule(i, 3, 3e-4, .02)) for i in range(1, 4)]
            p.write_text(''.join(json.dumps(x)+'\n' for x in records))
            validate_log(p, 3, 16)
            for bad in [records[:-1], [dict(records[0], nll=float('nan'))]+records[1:],
                        [dict(records[0], step=2)]+records[1:],
                        [dict(records[0], lr=0.)]+records[1:]]:
                p.write_text(''.join(json.dumps(x)+'\n' for x in bad))
                with self.assertRaises(ValueError):
                    validate_log(p, 3, 16)

    def test_prep_requires_terminal_marker_and_exit(self):
        with tempfile.TemporaryDirectory() as d, patch.object(job, 'PREP', Path(d)):
            log = Path(d)/'shell.log'
            (Path(d)/'complete.json').write_text('{}')
            (Path(d)/'gpus_after.json').write_text('[]')
            log.write_text(job.PREP_MARKER)
            with patch.object(job, 'prep_alive', return_value=True):
                self.assertFalse(job.prep_finished(log))
            with patch.object(job, 'prep_alive', return_value=False):
                self.assertTrue(job.prep_finished(log))
                log.write_text('incomplete')
                self.assertFalse(job.prep_finished(log))

    def test_live_child_never_recovered(self):
        w = job.Workflow()
        w.child = Mock()
        w.child.poll.return_value = None
        w.record = Mock()
        with patch.object(job, 'snapshot') as snap, patch.object(job, 'stop') as stop:
            w.recover_idle()
            snap.assert_not_called()
            stop.assert_not_called()

    def test_pipeline_order(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'fresh'
            events = []
            w = job.Workflow()
            def run(name, command, indices=(), spare=()):
                events.append(('run', name, list(indices), list(spare)))
            def record(**kw):
                (out/'status.json').write_text('{}')
                if kw.get('success'):
                    (out/'STOP_IDLE_WATCH').touch()
            w.run = run
            w.record = record
            w.burns = lambda g: events.append(('burn', g))
            w.free_after_wait = lambda g: events.append(('wait_free', g))
            with patch.object(job, 'OUT', out), patch.object(job, 'PROJECT', Path.cwd()), \
                 patch.object(job, 'PYTHON', sys.executable), patch.dict(os.environ, CONDA_DEFAULT_ENV='train_env'), \
                 patch.object(job, 'preflight'), patch.object(job, 'prep_finished', return_value=True), \
                 patch.object(job, 'stop', side_effect=lambda g: events.append(('stop', g))):
                w.execute(Path(d)/'prep.log')
            self.assertEqual(events, [
                ('run', 'preflight', [], []), ('run', 'validate_data', [], []),
                ('stop', job.ALL), ('wait_free', job.ALL), ('wait_free', job.ALL),
                ('run', 'train_common', job.ALL, []), ('wait_free', job.ALL),
                ('run', 'validate_common', [], []), ('burn', job.ALL),
                ('stop', job.ALL), ('wait_free', job.ALL), ('burn', job.SPARE),
                ('run', 'compile_tables', [0], job.SPARE), ('wait_free', [0]),
                ('run', 'validate_tables', [], []), ('stop', job.SPARE),
                ('wait_free', job.ALL), ('burn', job.ALL)])
            self.assertTrue((out/'complete.json').is_file())

    def test_failed_data_never_stops_burn_or_trains(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'fresh'
            w = job.Workflow()
            w.run = Mock(side_effect=[None, RuntimeError('bad data')])
            with patch.object(job, 'OUT', out), patch.object(job, 'PROJECT', Path.cwd()), \
                 patch.object(job, 'PYTHON', sys.executable), patch.dict(os.environ, CONDA_DEFAULT_ENV='train_env'), \
                 patch.object(job, 'preflight'), patch.object(job, 'prep_finished', return_value=True), \
                 patch.object(job, 'stop') as stop:
                with self.assertRaisesRegex(RuntimeError, 'bad data'):
                    w.execute(Path(d)/'prep.log')
                stop.assert_not_called()
                self.assertFalse((out/'complete.json').exists())

    def test_full_budget_commands(self):
        cmd = job.train_command()
        self.assertIn('--nproc_per_node=8', cmd)
        self.assertNotIn('--engineering', cmd)
        self.assertNotIn('--max-steps', cmd)
        self.assertEqual(cmd[cmd.index('--model-config')+1], str(job.ASSETS))
        self.assertEqual(cmd[cmd.index('--microbatch-segments')+1], '8')
        self.assertIn(str(job.OUT/'common/checkpoint-15259'), job.compile_command())


class StopSafety(unittest.TestCase):
    def result(self):
        ident = (20, 'start', [b'--multiprocessing-fork'])
        return [], {30: ident, 31: ident}, 20, (10, 'parent', [gpu.BURN.encode()])

    def test_pidfds_only_after_all_checks(self):
        result = self.result()
        events = []
        with patch.object(gpu, 'inspect', return_value=result), \
             patch.object(gpu, 'identity', return_value=result[1][30]), \
             patch.object(gpu.os, 'pidfd_open', side_effect=lambda p: events.append(('open', p)) or p+100, create=True), \
             patch.object(gpu.signal, 'pidfd_send_signal', side_effect=lambda fd, sig: events.append(('signal', fd, sig)), create=True), \
             patch.object(gpu.os, 'close') as close:
            gpu.stop([0, 1])
            self.assertEqual(events[:2], [('open', 30), ('open', 31)])
            self.assertEqual(events[2:], [('signal', 130, signal.SIGKILL), ('signal', 131, signal.SIGKILL)])
            self.assertEqual(close.call_count, 2)

    def test_pid_reuse_refuses_signaling(self):
        with patch.object(gpu, 'inspect', return_value=self.result()), \
             patch.object(gpu, 'identity', return_value=(20, 'CHANGED', [])), \
             patch.object(gpu.os, 'pidfd_open', return_value=130, create=True), \
             patch.object(gpu.signal, 'pidfd_send_signal', create=True) as send, patch.object(gpu.os, 'close'):
            with self.assertRaises(RuntimeError):
                gpu.stop([0, 1])
            send.assert_not_called()

    def test_pid1_refused(self):
        with self.assertRaises(RuntimeError):
            gpu.identity(1)

    def test_unknown_launcher_refused(self):
        cards = [dict(index=i, pids=[30+i], utilization_percent=98) for i in range(8)]
        def ident(pid):
            return (20, 'start', [b'--multiprocessing-fork']) if pid >= 30 else (2, 'p', [b'training.py'])
        with patch.object(gpu, 'preflight'), patch.object(gpu, 'snapshot', return_value=cards), \
             patch.object(gpu, 'identity', side_effect=ident):
            with self.assertRaisesRegex(RuntimeError, 'Not the original'):
                gpu.inspect(list(range(8)))


if __name__ == '__main__':
    unittest.main()
