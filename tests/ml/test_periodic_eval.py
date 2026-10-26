"""Periodic validation must leave native training updates and resume untouched."""
import json
import random
from pathlib import Path
import tempfile
import unittest

import numpy as np
from transformers import TrainerCallback
from transformers.trainer_utils import IntervalStrategy

import test_stopping as stopping
from test_stopping import torch
from test_document_isolation import row
from packed_evaluation import evaluate_packed
from resume_recipe import build_training_recipe, require_resume_training
from train import SaveConfigCallback


ROWS = [row([3, 4, 31, 8, 9, 10, 31], [0, 0, 0, 1, 1, 1, 1]),
        row([31] * 7, list(range(7))),
        row([3, 4, 5, 6, 7, 8, 31], [0] * 7)]


class RandomEvalCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        # Even evaluation integrations must not perturb training RNG.
        random.random()
        np.random.random()
        torch.rand(2)


def make_trainer(root, stop, validation=True):
    trainer, trace = stopping.StoppingTests().trainer(root, stop)
    if validation:
        trainer.args.eval_strategy = IntervalStrategy.STEPS
        trainer.args.do_eval = True
        trainer.args.eval_steps = 2
        trainer.args.eval_on_start = True
        trainer.args.per_device_eval_batch_size = 2
        trainer.eval_dataset = ROWS
        trainer.add_callback(RandomEvalCallback())
    trainer.training_recipe = build_training_recipe(trainer.model, trainer.args, 'sdpa', trainer.accelerator)
    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, SaveConfigCallback):
            callback.training_recipe = trainer.training_recipe
    return trainer, trace


def assert_training_equal(helper, a, b, trace_a, trace_b):
    helper.assertEqual(a.seen, b.seen)
    helper.assertEqual(trace_a.values, trace_b.values)
    helper.assert_nested_equal(a.model.state_dict(), b.model.state_dict())
    helper.assert_nested_equal(a.optimizer.state_dict(), b.optimizer.state_dict())
    helper.assert_nested_equal(a.lr_scheduler.state_dict(), b.lr_scheduler.state_dict())


class PeriodicEvalTests(unittest.TestCase):
    def test_monitoring_preserves_training_and_native_resume(self):
        helper = stopping.StoppingTests()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plain, plain_lr = make_trainer(root / 'plain', 8, False)
            plain.train()
            monitored, monitored_lr = make_trainer(root / 'monitored', 8)
            monitored.train()
            assert_training_equal(helper, plain, monitored, plain_lr, monitored_lr)
            entries = [x for x in monitored.state.log_history if 'eval_loss' in x]
            self.assertEqual([x['step'] for x in entries], [0, 2, 4, 6, 8])
            self.assertTrue(all(x['eval_num_tokens'] == 11 and x['eval_num_sequences'] == 3 for x in entries))
            expected = evaluate_packed(monitored.model, ROWS, monitored.data_collator, monitored.accelerator, 1)
            self.assertAlmostEqual(entries[-1]['eval_loss'], expected['loss'], places=6)
            first, first_lr = make_trainer(root / 'split', 3)
            first.train()
            resumed, resumed_lr = make_trainer(root / 'split', 8)
            checkpoint = root / 'split/checkpoint-3'
            require_resume_training(checkpoint, resumed.training_recipe, resumed.args)
            resumed.train(resume_from_checkpoint=str(checkpoint))
            self.assertEqual(first.seen + resumed.seen, monitored.seen)
            self.assertEqual(first_lr.values + resumed_lr.values, monitored_lr.values)
            helper.assert_nested_equal(resumed.model.state_dict(), monitored.model.state_dict())
            helper.assert_nested_equal(resumed.optimizer.state_dict(), monitored.optimizer.state_dict())
            helper.assert_nested_equal(resumed.lr_scheduler.state_dict(), monitored.lr_scheduler.state_dict())
            states = [torch.load(root / name / 'checkpoint-8/rng_state.pth', weights_only=False)
                      for name in ('plain', 'monitored', 'split')]
            for state in states[1:]:
                helper.assert_nested_equal(states[0]['cpu'], state['cpu'])
                self.assertEqual(states[0]['python'], state['python'])
                np.testing.assert_equal(states[0]['numpy'], state['numpy'])

    def test_metrics_empty_targets_mode_rng_and_gradients(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer, _ = make_trainer(folder, 8)
            trainer.model.train()
            # Keep deliberately mixed modes and an existing gradient.
            trainer.model.model.layers[0].eval()
            modes = [m.training for m in trainer.model.modules()]
            parameter = next(trainer.model.parameters())
            parameter.grad = torch.ones_like(parameter)
            gradient = parameter.grad.clone()
            buffers = {k: v.clone() for k, v in trainer.model.named_buffers()}
            before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
            metrics = trainer.evaluate()
            self.assertEqual(metrics['eval_num_tokens'], 11)
            self.assertEqual(metrics['eval_num_sequences'], 3)
            self.assertEqual(modes, [m.training for m in trainer.model.modules()])
            torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
            for k, v in trainer.model.named_buffers():
                torch.testing.assert_close(v, buffers[k], rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'no scored targets'):
                trainer.evaluate(ROWS[1:2])
            self.assertEqual(modes, [m.training for m in trainer.model.modules()])
            self.assertEqual(before[0], random.getstate())
            np.testing.assert_equal(before[1], np.random.get_state())
            torch.testing.assert_close(before[2], torch.get_rng_state(), rtol=0, atol=0)

    def test_unsupported_evaluation_modes_fail_clearly(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer, _ = make_trainer(folder, 8)
            for field in ('bf16_full_eval', 'fp16_full_eval', 'torch_compile', 'load_best_model_at_end'):
                setattr(trainer.args, field, True)
                with self.assertRaises(ValueError):
                    trainer.evaluate()
                setattr(trainer.args, field, False)
            from transformers.trainer_utils import SchedulerType
            for scheduler in (SchedulerType.REDUCE_ON_PLATEAU, SchedulerType.GREEDY):
                trainer.args.lr_scheduler_type = scheduler
                with self.assertRaisesRegex(ValueError, 'LR schedule'):
                    trainer.evaluate()


if __name__ == '__main__':
    unittest.main()
