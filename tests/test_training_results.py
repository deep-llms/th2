"""Offline artifact/queue checks; no ML dependencies, GPU or network."""
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import run_experiments as runner
from scripts import training_results as results

ROOT = Path(__file__).resolve().parents[1]


class TrainingResultTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'training'
        self.output.mkdir()

    def write(self, path, data):
        path.write_text(json.dumps(data))

    def stage(self, step=2, start=0, world=1):
        checkpoint = self.output / f'checkpoint-{step}'
        checkpoint.mkdir()
        recipe = dict(training=dict(max_steps=8), execution=dict(world_size=world,
            device_type='cpu', mixed_precision='no', train_batch_size_per_process=2))
        state = dict(global_step=step, max_steps=8, train_batch_size=2, log_history=[dict(
            step=step, eval_loss=2., eval_perplexity=math.exp(2), eval_num_tokens=3,
            eval_num_sequences=1, eval_nll_sum=6.)])
        config = dict(training=dict(max_steps=8, stop_after=step,
            resume_from_checkpoint=str(self.output / f'checkpoint-{start}') if start else None))
        for name, content in [('training_recipe.json', recipe), ('data_recipe.json', {}),
                              ('train_config.json', config), ('trainer_state.json', state)]:
            self.write(checkpoint / name, content)
            self.write(self.output / name, content)
        for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json'):
            self.write(checkpoint / name, {})
        for name in ('model.safetensors', 'optimizer.pt', 'scheduler.pt'):
            (checkpoint / name).write_bytes(b'fixture: completeness only, no tensor loading')
        names = ['rng_state.pth'] if world == 1 else [f'rng_state_{i}.pth' for i in range(world)]
        for name in names:
            (checkpoint / name).write_bytes(b'rng')
        self.write(self.output / 'train_results.json', dict(global_step=step, schedule_steps=8,
            train_runtime=1.25, train_loss=2.1))
        return checkpoint

    def evaluation(self, checkpoint):
        weight = checkpoint / 'model.safetensors'
        data = dict(status='ok', checkpoint=str(checkpoint), global_step=4,
            training_recipe_sha256=results.digest({}), config_sha256=hashlib.sha256(b'{}').hexdigest(),
            weights=[dict(path=str(weight), bytes=weight.stat().st_size, mtime_ns=weight.stat().st_mtime_ns)],
            loss=2., perplexity=math.exp(2), num_tokens=3, num_sequences=1, nll_sum=6.,
            attention_backend='sdpa', precision='fp32')
        self.write(self.root / 'evaluation.json', data)
        return self.root / 'evaluation.json'

    def test_stage_summary_and_frozen_checkpoint_evidence(self):
        self.stage()
        first = results.check_training(self.output, 2, 8)
        self.write(self.root / 'first.json', first)
        checkpoint = self.stage(4, 2)
        second = results.check_training(self.output, 4, 8, 2)
        self.write(self.root / 'second.json', second)
        paths = [self.root / 'first.json', self.root / 'second.json']
        evaluation = self.evaluation(checkpoint)
        summary = results.summarize(paths, evaluation)
        self.assertEqual(summary['total_stage_train_seconds'], 2.5)
        self.assertEqual([s['step'] for s in summary['stages']], [2, 4])
        self.assertIn('2 → 4', results.markdown(summary))
        # Same-size metadata changes must be caught, not just missing files.
        path = self.output / 'checkpoint-2/config.json'
        path.write_text('[]')
        with self.assertRaisesRegex(ValueError, 'Checkpoint changed'):
            results.summarize(paths, evaluation)

    def test_step_schedule_missing_state_and_bad_metrics_fail(self):
        checkpoint = self.stage()
        for step, schedule in [(2, 9), (4, 8)]:
            with self.assertRaises((ValueError, FileNotFoundError)):
                results.check_training(self.output, step, schedule)
        (checkpoint / 'optimizer.pt').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing file'):
            results.check_training(self.output, 2, 8)
        (checkpoint / 'optimizer.pt').write_bytes(b'x')
        path = self.output / 'train_results.json'
        data = json.loads(path.read_text())
        data['train_loss'] = float('nan')
        self.write(path, data)
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            results.check_training(self.output, 2, 8)

    def test_distributed_rng_shards_and_symlinks(self):
        checkpoint = self.stage(world=2)
        results.check_training(self.output, 2, 8)
        (checkpoint / 'rng_state_1.pth').unlink()
        with self.assertRaisesRegex(ValueError, 'rng_state_1'):
            results.check_training(self.output, 2, 8)
        (checkpoint / 'rng_state_1.pth').write_bytes(b'x')
        (checkpoint / 'model.safetensors').unlink()
        self.write(checkpoint / 'model.safetensors.index.json', dict(weight_map=dict(w='shard.safetensors')))
        with self.assertRaisesRegex(ValueError, 'shard.safetensors'):
            results.check_training(self.output, 2, 8)
        external = self.root / 'external'
        external.write_bytes(b'x')
        (checkpoint / 'shard.safetensors').symlink_to(external)
        with self.assertRaisesRegex(ValueError, 'escaping path'):
            results.check_training(self.output, 2, 8)
        (checkpoint / 'shard.safetensors').unlink()
        (checkpoint / 'shard.safetensors').write_bytes(b'x')
        results.check_training(self.output, 2, 8)

    def test_summary_rejects_wrong_eval_and_noncontiguous_stages(self):
        self.stage()
        first = results.check_training(self.output, 2, 8)
        self.write(self.root / 'first.json', first)
        checkpoint = self.stage(4, 2)
        second = results.check_training(self.output, 4, 8, 2)
        self.write(self.root / 'second.json', second)
        paths = [self.root / 'first.json', self.root / 'second.json']
        evaluation = self.evaluation(checkpoint)
        data = json.loads(evaluation.read_text())
        for key, value in [('global_step', 3), ('checkpoint', str(self.output / 'checkpoint-2')),
                           ('training_recipe_sha256', 'wrong'), ('perplexity', 1.0), ('num_tokens', 0)]:
            self.write(evaluation, dict(data, **{key: value}))
            with self.assertRaises(ValueError):
                results.summarize(paths, evaluation)
        self.write(evaluation, data)
        second['start_step'] = 1
        self.write(paths[1], second)
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            results.summarize(paths, evaluation)

    def test_failed_validator_stops_runner_before_later_job(self):
        checkpoint = self.stage()
        (checkpoint / 'scheduler.pt').unlink()
        jobs = [dict(name='check', argv=['{python}', str(ROOT / 'scripts/training_results.py'),
            'check', '--output-dir', str(self.output), '--expected-step', '2', '--schedule-steps', '8',
            '--output', '{run_dir}/checked.json'], required_outputs=[dict(path='checked.json')]),
            dict(name='later', argv=['{python}', str(ROOT / 'scripts/example_job.py'), '--output',
                '{run_dir}/later.json'], required_outputs=[dict(path='later.json')])]
        run = self.root / 'queue'
        with self.assertRaisesRegex(RuntimeError, 'exited with code 1'):
            runner.run_jobs(jobs, ROOT, run)
        self.assertFalse((run / 'complete.json').exists())
        self.assertFalse((run / 'later.log').exists())
        self.assertFalse((run / 'checked.json').exists())

    def test_example_and_no_overwrite(self):
        jobs = runner.load_jobs(ROOT / 'jobs.training.example.json')
        self.assertEqual(len(jobs), 6)
        self.assertTrue(all('gpus' not in job for job in jobs))
        output = self.root / 'receipt.json'
        results.publish(output, 'original')
        with self.assertRaises(FileExistsError):
            results.publish(output, 'replacement')
        self.assertEqual(output.read_text(), 'original')


if __name__ == '__main__':
    unittest.main()
