"""Mock-only orchestration checks. Never signal or query actual GPU workers."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import pilot_stage2 as job
from validate_stage2 import stage2_log, fresh_reader_hash
from ccm.cli import parser
from ccm.contracts import schedule, PILOT
from ccm.model import MemoryLM
from ccm.runtime import MasterAdamW, optimizer_groups
from transformers import Qwen3Config
import torch
from validate_stage2 import optimizer_layout, validate_optimizer


class Stage2Tests(unittest.TestCase):
    def test_commands(self):
        hashes = [fresh_reader_hash(), fresh_reader_hash()]
        self.assertEqual(*hashes)
        for arm in job.ARMS:
            cmd = job.train_command(arm)
            self.assertIn('--nproc_per_node=8', cmd)
            args = parser().parse_args(cmd[cmd.index('train'):])
            self.assertEqual((args.phase, args.arm, args.seed), ('stage2', arm, 17))
            self.assertEqual(args.checkpoint, str(job.COMMON/'common/checkpoint-15259'))
            self.assertFalse(args.engineering or args.online)
            self.assertEqual(args.table, None if arm in ('base', 'grad') else str(job.COMMON/'tables'/arm))
        for arm in job.ARMS:
            cmd = job.eval_command(arm)
            args = parser().parse_args(cmd[cmd.index('evaluate'):])
            self.assertEqual(args.role, 'dev')
            self.assertFalse(args.final_evaluation)
            self.assertEqual(args.diagnostic_table, str(job.COMMON/'tables/contextual'))
            self.assertEqual(args.checkpoint, str(job.OUT/'train'/arm/'checkpoint-3815'))

    def test_optimizer_layout_and_snapshot(self):
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                            num_attention_heads=2, num_key_value_heads=1, head_dim=8, tie_word_embeddings=True)
        config._attn_implementation = 'sdpa'
        for arm in job.ARMS:
            with patch('validate_stage2.PILOT', slots=4):
                layout = optimizer_layout(config, arm)
            table = None if arm in ('base', 'grad') else torch.zeros(4, 16)
            m = MemoryLM(config, arm, slots=4, table=table).bfloat16()
            m.set_phase('stage2')
            opt = MasterAdamW(optimizer_groups(m, 'stage2'))
            for g in opt.inner.param_groups:
                g['lr'] = 1.5e-5*g['multiplier']
            for p in m.parameters():
                p.grad = torch.ones_like(p)
            opt.step()
            state = opt.state_dict()
            for v in state['optimizer']['state'].values():
                v['step'] = torch.tensor(3815.)
            self.assertEqual(validate_optimizer(state, layout, m.state_dict()),
                             sum(p.numel() for p in m.parameters()))
            state['masters'][0].fill_(float('nan'))
            with self.assertRaises(ValueError):
                validate_optimizer(state, layout, m.state_dict())

    def test_log_gate(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'train.jsonl'
            rows = [dict(step=i, input_tokens=i*PILOT.batch_tokens, nll=2., grad_norm=.1,
                         lr=schedule(i, 3815, 1.5e-4, .02)) for i in range(1, 3816)]
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            stage2_log(path)
            for bad in (rows[:-1], [dict(rows[0], lr=3e-4)]+rows[1:],
                        [dict(rows[0], nll=float('nan'))]+rows[1:]):
                path.write_text(''.join(json.dumps(r)+'\n' for r in bad))
                with self.assertRaises(ValueError):
                    stage2_log(path)

    def test_sequence_and_completion(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'fresh'
            w = job.Stage2()
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
            self.assertEqual(evals, ['eval_'+a for a in job.ARMS])
            self.assertEqual(events[-1], ('validate', 'panel', None))
            self.assertTrue((out/'complete.json').is_file())

    def test_failed_inputs_never_reclaims_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'fresh'
            w = job.Stage2()
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

    def test_failed_training_blocks_following_arms(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'fresh'
            w = job.Stage2()
            w.validate = Mock()
            w.disarm_old_observer = Mock()
            w.free_after_wait = Mock()
            w.run = Mock(side_effect=RuntimeError('training failed'))
            w.burns = Mock()
            with patch.object(job, 'OUT', out), patch.object(job, 'PROJECT', Path.cwd()), \
                 patch.dict(os.environ, CONDA_DEFAULT_ENV='train_env'), patch.object(job, 'preflight'), \
                 patch.object(job, 'inspect'), patch.object(job, 'stop'):
                with self.assertRaisesRegex(RuntimeError, 'training failed'):
                    w.execute()
            self.assertEqual(w.run.call_count, 1)
            w.burns.assert_not_called()
            self.assertFalse((out/'complete.json').exists())

    def test_disarm_correct_previous_observer_without_signals(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'complete.json').write_text(json.dumps(dict(success=True, stage='complete')))
            w = job.Stage2()
            w.record = Mock()
            ident = (123, '456', [b'python', b'scripts/pilot_stage1.py'])
            with patch.object(job, 'PREVIOUS', root), patch.object(job.subprocess, 'run', return_value=Mock(returncode=0)), \
                 patch.object(job.subprocess, 'check_output', return_value='0 321'), \
                 patch.object(job, 'identity', side_effect=[ident, ProcessLookupError()]), patch.object(job, 'stop') as stop:
                w.disarm_old_observer()
                self.assertTrue((root/'STOP_IDLE_WATCH').is_file())
                stop.assert_not_called()


if __name__ == '__main__':
    unittest.main()
