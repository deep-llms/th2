"""Actual Trainer fork/resume, optimizer/RNG/data and linear-LR boundary tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file

from deep_kv.cooldown import stage, schedule, transition, validate
from deep_kv.proxy_training import ProxyTrainer
from tests.test_proxy_training import proxy_fixture
from tests.test_train import invoke
from scripts.check_optimized_resume import state_hash
from scripts.stage_deep_kv_resume import digest


class CooldownTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_schedule_boundaries_and_validation(self):
        original = lambda step: .9 - step * .01
        f = schedule(original, 10, 4)
        for actual, expected in zip([f(s) for s in (9, 10, 11, 12, 14, 15)], [.81, .8, .6, .4, 0., 0.]):
            self.assertAlmostEqual(actual, expected)
        for start, steps in ((None, 4), (10, None), (0, 4), (10, 0), (10, 11), (True, 2)):
            with self.assertRaises(ValueError): validate(start, steps, 20, 2)

    def test_real_fork_interrupted_resume_exact_states_and_batches(self):
        for arm in ('A', 'P6-iso'):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = proxy_fixture(root)
                from tests.test_proxy_memory import config as model_config
                model_config().save_pretrained(root / 'model')
                config.update(arm=arm, proxy_groups=2, max_steps=5, save_total_limit=0,
                              eval_steps=5, checkpoint_layers=False, checkpoint_aux=False, checkpoint_lm=False)
                source = root / 'source'
                invoke(root, {**config, 'output_dir': str(source), 'stop_after': 2})
                checkpoint = source / 'checkpoint-2'
                original = {p.name: digest(p) for p in checkpoint.iterdir() if p.is_file()}
                captures = {}
                for name in ('whole', 'split'):
                    output = root / name
                    stage(checkpoint, output, 3, 2)
                    captures[name] = []
                    step = ProxyTrainer.training_step
                    def observed(self, model, inputs, *a, **kw):
                        captures[name].append((self.state.global_step, state_hash(inputs),
                                               [g['lr'] for g in self.optimizer.param_groups]))
                        return step(self, model, inputs, *a, **kw)
                    args = {**config, 'output_dir': str(output), 'cooldown_start': 3, 'cooldown_steps': 2}
                    with patch.object(ProxyTrainer, 'training_step', observed):
                        if name == 'split': invoke(root, {**args, 'stop_after': 4})
                        invoke(root, {**args, 'stop_after': 5})
                    scheduler = torch.load(output / 'checkpoint-5/scheduler.pt', weights_only=True)
                    self.assertEqual(scheduler['last_epoch'], 5)
                    self.assertTrue(all(lr == 0 for lr in scheduler['_last_lr']))
                    self.assertTrue(all(v is None for v in scheduler['lr_lambdas']))
                self.assertEqual(captures['whole'], captures['split'])
                for name in ('model.safetensors', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
                    def load(folder):
                        p = root / folder / 'checkpoint-5' / name
                        return load_file(p) if name.endswith('safetensors') else torch.load(p, weights_only=False)
                    self.assertEqual(state_hash(load('whole')), state_hash(load('split')), name)
                self.assertEqual(original, {p.name: digest(p) for p in checkpoint.iterdir() if p.is_file()})
                # Cooldown is an explicit schedule fork; it must not be silently retimed.
                with self.assertRaisesRegex(ValueError, 'existing cooldown'):
                    invoke(root, {**config, 'output_dir': str(root / 'whole'),
                                  'cooldown_start': 2, 'cooldown_steps': 3})
                self.assertNotIn('cooldown_start', json.loads((source / 'train_config.json').read_text())['pilot'])

    def test_fork_provenance_rejects_unstaged_schedule(self):
        previous = {'pilot': {}}
        requested = {'pilot': {'cooldown_start': 10, 'cooldown_steps': 2}}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError): transition(previous, requested, tmp, tmp)


if __name__ == '__main__': unittest.main()
