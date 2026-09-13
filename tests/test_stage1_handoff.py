"""Mock-only orchestration checks. Never signal or query actual GPU workers."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import pilot_stage1 as job
from validate_stage1 import stage1_log, fresh_reader_hash
from ccm.cli import parser
from ccm.contracts import schedule, PILOT


class Stage1Tests(unittest.TestCase):
    def test_commands(self):
        hashes = [fresh_reader_hash(), fresh_reader_hash()]
        self.assertEqual(*hashes)
        for arm in job.ARMS:
            cmd = job.train_command(arm)
            self.assertIn('--nproc_per_node=8', cmd)
            args = parser().parse_args(cmd[cmd.index('train'):])
            self.assertEqual((args.phase, args.arm, args.seed), ('stage1', arm, 17))
            self.assertEqual(args.checkpoint, str(job.COMMON/'common/checkpoint-15259'))
            self.assertFalse(args.engineering or args.online)
            self.assertEqual(args.table, None if arm == 'grad' else str(job.COMMON/'tables'/arm))
        for arm in ('base',)+job.ARMS:
            cmd = job.eval_command(arm)
            args = parser().parse_args(cmd[cmd.index('evaluate'):])
            self.assertEqual(args.role, 'dev')
            self.assertFalse(args.final_evaluation)
            self.assertEqual(args.diagnostic_table, str(job.COMMON/'tables/contextual'))

    def test_log_gate(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'train.jsonl'
            rows = [dict(step=i, input_tokens=i*PILOT.batch_tokens, nll=2., grad_norm=.1,
                         lr=schedule(i, 977, 5e-4, .05)) for i in range(1, 978)]
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            stage1_log(path)
            for bad in (rows[:-1], [dict(rows[0], lr=3e-4)]+rows[1:],
                        [dict(rows[0], nll=float('nan'))]+rows[1:]):
                path.write_text(''.join(json.dumps(r)+'\n' for r in bad))
                with self.assertRaises(ValueError):
                    stage1_log(path)

    def test_sequence_and_completion(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'fresh'
            w = job.Stage1()
            events = []
            def record(**fields):
                (out/'status.json').write_text(json.dumps(fields))
                if fields.get('success'):
                    (out/'STOP_IDLE_WATCH').touch()
            w.record = record
            w.run = lambda name, cmd, indices=(), spare=(): events.append(('run', name, list(indices), list(spare)))
            w.validate = lambda stage, arm=None, spare=(): events.append(('validate', stage, arm))
            w.disarm_old_observer = lambda: events.append(('disarm',))
            w.burns = lambda g: events.append(('burn', g))
            w.free_after_wait = lambda g: events.append(('wait_free', g))
            with patch.object(job, 'OUT', out), patch.object(job, 'PROJECT', Path.cwd()), \
                 patch.dict(os.environ, CONDA_DEFAULT_ENV='train_env'), patch.object(job, 'preflight'), \
                 patch.object(job, 'inspect'), patch.object(job, 'snapshot', return_value=[]), \
                 patch.object(job, 'stop', side_effect=lambda g: events.append(('stop', g))):
                w.execute()
            self.assertEqual(events[:5], [('validate', 'inputs', None), ('disarm',),
                              ('stop', job.ALL), ('wait_free', job.ALL), ('wait_free', job.ALL)])
            for i, arm in enumerate(job.ARMS):
                self.assertEqual(events[5+i*3:8+i*3], [('run', 'train_'+arm, job.ALL, []),
                                                      ('wait_free', job.ALL), ('validate', 'train', arm)])
            evals = [e[1] for e in events if e[0] == 'run' and e[1].startswith('eval_')]
            self.assertEqual(evals, ['eval_'+a for a in ('base',)+job.ARMS])
            self.assertEqual(events[-1], ('validate', 'panel', None))
            self.assertTrue((out/'complete.json').is_file())

    def test_failed_inputs_never_reclaims_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'fresh'
            w = job.Stage1()
            w.validate = Mock(side_effect=ValueError('wrong artifact'))
            w.disarm_old_observer = Mock()
            with patch.object(job, 'OUT', out), patch.object(job, 'PROJECT', Path.cwd()), \
                 patch.dict(os.environ, CONDA_DEFAULT_ENV='train_env'), patch.object(job, 'preflight'), \
                 patch.object(job, 'stop') as stop:
                with self.assertRaises(ValueError):
                    w.execute()
                stop.assert_not_called()
                w.disarm_old_observer.assert_not_called()
                self.assertFalse((out/'complete.json').exists())


if __name__ == '__main__':
    unittest.main()
