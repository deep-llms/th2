"""Regression tests for prior preservation, curriculum evaluation and LM logging."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch
from datasets import Dataset
from transformers import TrainerCallback

from test_model import model, batch, mapped_pairs
from bdot.model import RouteContext
from bdot.trainer import BDOTTrainer
from packed_training import PackedCollator
from packed_evaluation import evaluate_packed
from train import BaselineTrainingArguments, StopAfterCallback


ROWS = [dict(input_ids=[2,3,4,5,6,7], labels=[2,3,4,5,6,7], attention_mask=[1]*6,
             segments=[0,0,1,1,1,1]),
        dict(input_ids=[3,4,5,6,2,8], labels=[3,4,5,6,2,8], attention_mask=[1]*6,
             segments=[0]*6)]


class RecordSums(TrainerCallback):
    def __init__(self):
        self.sums = []

    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Runs before RoutingUpdateCallback drains the per-update accumulators.
        self.sums.append(model.metric_totals.clone())


def trainer(root, *, curriculum='A', stop=None, validation=False):
    m = model(curriculum=curriculum)
    args = BaselineTrainingArguments(output_dir=str(root), use_cpu=True, report_to='none',
        max_steps=20, stop_after=stop, save_steps=5, logging_steps=5,
        per_device_train_batch_size=1, gradient_accumulation_steps=2,
        per_device_eval_batch_size=1, learning_rate=.001, max_grad_norm=0.,
        weight_decay=.1, remove_unused_columns=False, disable_tqdm=True,
        train_sampling_strategy='sequential', seed=5, data_seed=0, dataloader_pin_memory=False,
        do_eval=validation, eval_strategy='steps' if validation else 'no', eval_steps=1)
    sums = RecordSums()
    return BDOTTrainer(model=m, args=args, train_dataset=Dataset.from_list(ROWS*24),
        eval_dataset=ROWS, data_collator=PackedCollator(), callbacks=[sums, StopAfterCallback()]), sums


class ScientificReviewTests(unittest.TestCase):
    def test_home_only_preserves_router_and_matches_native_even_with_optimizer_momentum(self):
        m = model(curriculum='A', step_film=False)
        m.progress = .05
        native = m.native_reference()
        dense = model(routing='fixed', curriculum='none', r=0, step_film=False)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=.1)
        native_opt = torch.optim.AdamW(native.parameters(), lr=3e-4, weight_decay=.1)
        dense_opt = torch.optim.AdamW(dense.parameters(), lr=3e-4, weight_decay=.1)
        routers = [p for layer in m.routed_steps for p in layer.router.parameters()]
        originals = [p.detach().clone() for p in routers]
        for p in routers:
            opt.state[p] = dict(step=torch.tensor(7.), exp_avg=torch.ones_like(p), exp_avg_sq=torch.ones_like(p))
        for _ in range(5):
            opt.zero_grad(set_to_none=True); native_opt.zero_grad(set_to_none=True); dense_opt.zero_grad(set_to_none=True)
            x = m(**batch()); y = native(**batch()); d = dense(**batch())
            torch.testing.assert_close(x.loss, y.loss, rtol=0, atol=0)
            torch.testing.assert_close(x.loss, d.loss, rtol=0, atol=0)
            x.loss.backward(); y.loss.backward(); d.loss.backward()
            self.assertTrue(all(p.grad is None for p in routers))
            self.assertEqual(float(m.metric_totals[2]), 0.)
            opt.step(); native_opt.step(); dense_opt.step(); m.finish_update(); dense.finish_update()
        for p, old in zip(routers, originals):
            torch.testing.assert_close(p, old, rtol=0, atol=0)
            self.assertEqual(float(opt.state[p]['step']), 7.)
        for p, q in mapped_pairs(m, native):
            # Native residual arithmetic differs by a few FP32 ULPs; same T1 tolerance.
            torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-6)
        for name, p in dense.named_parameters():
            torch.testing.assert_close(p, dict(m.named_parameters())[name], rtol=0, atol=0)
        m.eval(); m.capture_routes = True
        m(**batch())
        self.assertTrue((m.last_routes == torch.arange(4)[:, None, None]).all())

    def test_forced_no_reuse_rows_have_no_z_gradient(self):
        m = model(reuse=False, curriculum='none')
        step = m.routed_steps[1]
        states = step.transitions_1[0, torch.tensor([0, 1])]
        visits = torch.nn.functional.one_hot(torch.tensor([0, 1]), 4).float()
        ctx = RouteContext(visits, states, torch.Generator().manual_seed(1), 1, False, 0.,
                           torch.zeros(4), [], [], [], [])
        h = torch.randn(2, 1, 16, requires_grad=True)
        allowed = step.allowed_1[states]
        self.assertEqual((allowed.sum(-1) > 1).tolist(), [True, False])
        z = step.router(h.flatten(0,1), visits, allowed)
        expected = torch.logsumexp(z[0], -1).square()
        step.select(h, ctx)
        torch.testing.assert_close(ctx.z_losses[0], expected)
        ctx.z_losses[0].backward()
        self.assertEqual(float(h.grad[1].abs().sum()), 0.)
        self.assertGreater(float(h.grad[0].abs().sum()), 0.)

    def test_masked_z_loss_accumulation_matches_full_batch(self):
        a = model(reuse=False, curriculum='none'); b = copy.deepcopy(a)
        inputs = batch()
        count = inputs['labels'][:, 1:].ne(-100).sum()
        decisions = inputs['input_ids'].numel()*4
        # Some rows force a choice at step 1; every row forces the final step.
        paths = torch.tensor([[0, 1, 2, 3], [1, 0, 2, 3]]).T[:, :, None].expand(4, 2, 6)
        out = a(**inputs, bdot_forced_routes=paths, num_items_in_batch=count, bdot_num_decisions=decisions)
        out.loss.backward(); losses = []
        for i in range(2):
            part = b(**{k: v[i:i+1] for k, v in inputs.items()}, bdot_forced_routes=paths[:, i:i+1],
                     num_items_in_batch=count, bdot_num_decisions=decisions)
            part.loss.backward(); losses.append(part.loss.detach())
        torch.testing.assert_close(out.loss.detach(), sum(losses), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(a.metric_totals[2:4], b.metric_totals[2:4], rtol=1e-6, atol=1e-6)
        for p, q in zip(a.parameters(), b.parameters()):
            if p.grad is None:
                self.assertIsNone(q.grad)
            else:
                torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-6)
        self.assertTrue(all(p.grad is None for p in a.routed_steps[-1].router.parameters()))

    def test_dual_evaluation_policies_and_state_preservation(self):
        for curriculum in ('A', 'B'):
            with self.subTest(curriculum=curriculum), tempfile.TemporaryDirectory() as directory:
                t, _ = trainer(Path(directory), curriculum=curriculum)
                m = t.model
                m.progress = .05; m.train(); m.eval_counter = 17
                before = copy.deepcopy(m.state_dict())
                rng = torch.get_rng_state().clone()
                metrics = t.evaluate()
                self.assertEqual(metrics['eval_curriculum_reach'], 0 if curriculum == 'A' else 1)
                self.assertEqual(metrics['eval_curriculum_soft'], curriculum == 'B')
                self.assertEqual(metrics['eval_spike_loss'], metrics['eval_curriculum_loss'])
                self.assertTrue(m.training)
                self.assertEqual(m.evaluation_policy, 'target')
                self.assertEqual(m.eval_counter, 17)
                torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
                for k, v in m.state_dict().items():
                    torch.testing.assert_close(v, before[k], rtol=0, atol=0)
                m.evaluation_policy = 'curriculum'
                expected = evaluate_packed(m, ROWS, t.data_collator, t.accelerator)
                self.assertAlmostEqual(expected['loss'], metrics['eval_curriculum_loss'], places=7)
                m.evaluation_policy = 'target'
                final = t.evaluate(include_curriculum=False)
                self.assertNotIn('eval_curriculum_loss', final)
                self.assertAlmostEqual(final['eval_loss'], metrics['eval_loss'], places=7)

    def test_logging_sums_resume_before_widening_and_monitoring_do_not_change_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full, sums = trainer(root/'full'); full_result = full.train()
            monitored, _ = trainer(root/'monitored', validation=True); monitored.train()
            first, _ = trainer(root/'split', stop=1); first.train()
            resumed, _ = trainer(root/'split')
            resumed_result = resumed.train(resume_from_checkpoint=str(root/'split/checkpoint-1'))
            for candidate in (monitored, resumed):
                for k, v in full.model.state_dict().items():
                    torch.testing.assert_close(v, candidate.model.state_dict()[k], rtol=0, atol=0, msg=k)
                for k, state in full.optimizer.state_dict()['state'].items():
                    for name, value in state.items():
                        torch.testing.assert_close(value, candidate.optimizer.state_dict()['state'][k][name], rtol=0, atol=0)
            entries = [x for x in full.state.log_history if 'objective_loss' in x]
            self.assertEqual([x['step'] for x in entries], [5, 10, 15, 20])
            for i, entry in enumerate(entries):
                expected = torch.stack(sums.sums[5*i:5*i+5]).sum(0)
                self.assertAlmostEqual(entry['lm_loss'], float(expected[0]/expected[1]), places=6)
                self.assertAlmostEqual(entry['router_z_loss'], float(expected[2]/expected[3]), places=6)
                self.assertNotIn('loss', entry)
                self.assertGreater(entry['input_tokens_per_second_interval_wall'], 0.)
                self.assertIsNone(entry['peak_allocated_bytes'])
            self.assertNotIn('train_loss', full_result.metrics)
            self.assertIn('train_objective_loss', full_result.metrics)
            self.assertEqual(full_result.metrics['train_lm_loss'], resumed_result.metrics['train_lm_loss'])


if __name__ == '__main__':
    unittest.main()
