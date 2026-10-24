import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from run_experiments import load_jobs
from scripts.stop_optimized_launch import prepare_optimized, gate

ROOT = Path(__file__).resolve().parents[2]


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


class OptimizedQueueTests(unittest.TestCase):
    def test_fresh_pair_has_all_gates_and_unchanged_schedule_and_native_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); parent = root/'parent'; previous = root/'previous'; run = root/'run'
            run.mkdir()
            recipe = dict(max_steps=10000, block_size=1024, seed=0, data_seed=0,
                per_device_train_batch_size=16, gradient_accumulation_steps=4,
                train_sampling_strategy='sequential', dataloader_drop_last=True,
                attention_backend='fa4', gradient_checkpointing=False, save_steps=250,
                save_total_limit=None, output_dir=str(parent/'original'),
                lr_scheduler_type='cosine_with_min_lr', lr_scheduler_kwargs={'min_lr_rate': .1})
            write(parent/'prepared/D-s0.train.json', recipe)
            write(previous/'evaluation_manifest.json', {n: dict(checkpoint=str(previous/n/'checkpoint-1000'))
                  for n in ('D-continue','D-DualCE6')})

            def initialize(cp, dest, prob, *, stop_settings, exit_training, schedule_steps):
                self.assertEqual(cp, parent/'original/checkpoint-10000')
                self.assertEqual(schedule_steps, 10000)
                config = json.loads((ROOT/'configs/bdot/D.json').read_text())
                config['bdot_stop'] = stop_settings
                if exit_training is not None:
                    config['bdot_exit_training'] = exit_training
                write(dest/'config.json', config)

            with patch('scripts.stop_long_launch.initialize', side_effect=initialize):
                jobs = prepare_optimized(parent, previous, run, ROOT)
            load_jobs(run/'jobs.json')
            names = [j['name'] for j in jobs]
            self.assertEqual(len(jobs), 18)
            self.assertLess(names.index('optimization_gate'), names.index('D-continue-5000'))
            self.assertLess(names.index('eval_smoke'), names.index('D-continue-5000'))
            self.assertTrue(all(j['gpus'] == list(range(8)) for j in jobs))
            self.assertIn('--exit-shared-prefix', jobs[2]['argv'])
            for j in jobs:
                if j['name'] in ('optimization_ddp', 'resume_gate') or j['name'].startswith('benchmark-'):
                    self.assertEqual(j['argv'][j['argv'].index('--expert-backend')+1], 'grouped')
            cfg = json.loads((run/'prepared/D-DualCE6-init/config.json').read_text())
            self.assertEqual(cfg['bdot_exit_training'], dict(depth=6, ce_weight=.5, kl_weight=0.,
                                                           ce_backend='native', shared_prefix=True))
            resume = json.loads((run/'resume-model.json').read_text())
            self.assertEqual(cfg['bdot_exit_training'], resume['bdot_exit_training'])
            control = json.loads((run/'prepared/D-continue-init/config.json').read_text())
            self.assertNotIn('bdot_exit_training', control)
            for arm in ('D-continue', 'D-DualCE6'):
                a = json.loads((run/f'prepared/{arm}-5000.train.json').read_text())
                b = json.loads((run/f'prepared/{arm}-10000.train.json').read_text())
                self.assertNotIn('resume_from_checkpoint', a)
                self.assertEqual(b.pop('resume_from_checkpoint'), str(run/f'prepared/runs/{arm}/checkpoint-5000'))
                a.pop('stop_after'); b.pop('stop_after'); self.assertEqual(a, b)
                self.assertEqual((a['max_steps'], a['warmup_steps'], a['train_start_sequence']), (10000, 200, 5120000))
                self.assertIsNone(a['save_total_limit'])

    def test_throughput_gate_refuses_slow_or_wrong_workload(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for p, status in [('numerics.json','passed'), ('capacity.json','ok'),
                              ('resume/summary.json','passed'), ('ddp/rank0.json','passed'), ('ddp/rank1.json','passed')]:
                write(run/p, dict(status=status))
            write(run/'capacity.json', dict(status='ok', branches={'D': dict(device_total_bytes=1000)}))
            for case, time in [('control',1.), ('reference',3.), ('optimized',1.5)]:
                for repeat in (1,2):
                    write(run/f'benchmark-{case}-{repeat}.json', dict(status='completed', world_size=8,
                        settings=dict(batch_size=16, accumulation=4, length=1024, attention_backend='fa4',
                                      expert_backend='grouped', checkpointing=False),
                        seconds_per_update=time, peak_allocated_bytes=10))
            path = run/'benchmark-optimized-1.json'; data = json.loads(path.read_text())
            bad = copy.deepcopy(data); bad['seconds_per_update'] = 5.; write(path, bad)
            with self.assertRaisesRegex(ValueError, 'did not improve'): gate(run)
            self.assertFalse((run/'optimization-gate.json').exists())
            bad = copy.deepcopy(data); bad['world_size'] = 1; write(path, bad)
            with self.assertRaisesRegex(ValueError, 'Wrong throughput'): gate(run)
            write(path, data); gate(run)
            self.assertEqual(json.loads((run/'optimization-gate.json').read_text())['update_time_reduction'], .5)

    def test_gate_cli_runs_without_launch_only_modules(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source/scripts'
            source.mkdir(parents=True)
            for name in ('stop_optimized_launch.py', 'prepare_stop_adaptation.py'):
                shutil.copy2(ROOT / 'scripts' / name, source / name)
            run = root / 'run'
            for path, status in [('numerics.json', 'passed'), ('resume/summary.json', 'passed'),
                                 ('ddp/rank0.json', 'passed'), ('ddp/rank1.json', 'passed')]:
                write(run / path, dict(status=status))
            write(run / 'capacity.json', dict(status='ok', branches={'D': dict(device_total_bytes=1000)}))
            for case, seconds in [('control', 1.), ('reference', 3.), ('optimized', 1.5)]:
                for repeat in (1, 2):
                    write(run / f'benchmark-{case}-{repeat}.json', dict(status='completed', world_size=8,
                        settings=dict(batch_size=16, accumulation=4, length=1024, attention_backend='fa4',
                                      expert_backend='grouped', checkpointing=False),
                        seconds_per_update=seconds, peak_allocated_bytes=10))
            result = subprocess.run([sys.executable, '-I', str(source / 'stop_optimized_launch.py'),
                                     'gate', '--run', str(run)], cwd=root, capture_output=True, text=True,
                                    env={k: v for k, v in os.environ.items() if k != 'PYTHONPATH'})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads((run / 'optimization-gate.json').read_text())['status'], 'passed')


if __name__ == '__main__':
    unittest.main()
