"""Opt-in execution changes against the independent two-traversal reference."""
import copy
from dataclasses import replace
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_exit_training import dual
from test_model import batch
from bdot.exit_training import loss_sums
from bdot.model import BDOTForCausalLM


def optimized(model, *, native=True):
    model.exit_options = replace(model.exit_options, shared_prefix=True,
                                 ce_backend='native' if native else 'chunked')
    model.config.bdot_exit_training.update(shared_prefix=True,
                                          ce_backend=model.exit_options.ce_backend)
    model.exit_options.validate(model)
    return model


def assert_gradients(test, a, b):
    for name, p in a.named_parameters():
        q = b.get_parameter(name)
        if p.grad is None:
            test.assertIsNone(q.grad, name)
        else:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-4, msg=name)


class ExitOptimizationTests(unittest.TestCase):
    def test_native_ce_matches_masked_reference_and_empty_rows(self):
        for empty in (False, True):
            torch.manual_seed(83)
            a = torch.randn(2, 7, 113, requires_grad=True)
            b = a.detach().clone().requires_grad_()
            teacher = torch.randn_like(a, requires_grad=True)
            labels = torch.tensor([[-100, 2, 4, -100, 3, 7, 8], [-100]*7])
            if empty:
                labels.fill_(-100)
            old = loss_sums(a, teacher, labels, 3, False)
            new = loss_sums(b, teacher, labels, 3, False, ce_backend='native')
            torch.testing.assert_close(new, old, atol=2e-6, rtol=1e-6)
            old.sum().backward(); new.sum().backward()
            torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
            self.assertIsNone(teacher.grad)
        with self.assertRaises(ValueError):
            loss_sums(a, teacher, labels, include_kl=True, ce_backend='native')

    def test_shared_prefix_objective_gradients_statistics_and_updates(self):
        for partial in (False, True):
            for kl in (0., .5):
                a = dual(partial, kl); b = optimized(copy.deepcopy(a), native=kl == 0)
                a.capture_routes = b.capture_routes = True
                oa = torch.optim.AdamW(a.parameters(), lr=1e-4)
                ob = torch.optim.AdamW(b.parameters(), lr=1e-4)
                for _ in range(3):
                    x = a(**batch()); y = b(**batch())
                    torch.testing.assert_close(x.logits, y.logits, atol=2e-6, rtol=2e-5)
                    torch.testing.assert_close(x.loss, y.loss, atol=2e-6, rtol=1e-6)
                    x.loss.backward(); y.loss.backward()
                    assert_gradients(self, a, b)
                    torch.testing.assert_close(a.last_routes, b.last_routes, atol=0, rtol=0)
                    torch.testing.assert_close(a.stop_controller.last_depths,
                                               b.stop_controller.last_depths, atol=0, rtol=0)
                    for name in ('load_counts', 'metric_totals', 'exit_metric_totals'):
                        torch.testing.assert_close(getattr(a, name), getattr(b, name),
                                                   atol=2e-5, rtol=1e-6)
                    oa.step(); ob.step(); oa.zero_grad(); ob.zero_grad()
                    a.finish_update(); b.finish_update()
                    torch.testing.assert_close(a.balance_bias, b.balance_bias, atol=0, rtol=0)
                    self.assertEqual(int(a.routing_counter), int(b.routing_counter))
                for p, q in zip(a.parameters(), b.parameters()):
                    torch.testing.assert_close(p, q, atol=2e-6, rtol=1e-4)

    def test_accumulation_with_zero_local_targets(self):
        a = optimized(dual(kl=0)); b = copy.deepcopy(a)
        inputs = batch(); inputs['labels'][0].fill_(-100)
        count = inputs['labels'][:, 1:].ne(-100).sum()
        decisions = inputs['input_ids'].numel() * 4
        x = a(**inputs, num_items_in_batch=count, bdot_num_decisions=decisions)
        x.loss.backward(); total = 0.
        for row in range(2):
            y = b(**{k: v[row:row+1] for k, v in inputs.items()},
                  num_items_in_batch=count, bdot_num_decisions=decisions)
            y.loss.backward(); total += y.loss.detach()
        torch.testing.assert_close(x.loss.detach(), total, atol=2e-6, rtol=1e-6)
        assert_gradients(self, a, b)

    def test_checkpointing_and_save_reload(self):
        a = optimized(dual(kl=0)); b = copy.deepcopy(a)
        b.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        x = a(**batch()); y = b(**batch()); x.loss.backward(); y.loss.backward()
        torch.testing.assert_close(x.loss, y.loss, atol=0, rtol=0)
        assert_gradients(self, a, b)
        a.finish_update()
        with tempfile.TemporaryDirectory() as tmp:
            a.save_pretrained(tmp)
            b = BDOTForCausalLM.from_pretrained(tmp, attn_implementation='sdpa').train()
        self.assertTrue(b.exit_options.shared_prefix)
        self.assertEqual(b.exit_options.ce_backend, 'native')
        torch.testing.assert_close(a(**batch()).loss, b(**batch()).loss, atol=0, rtol=0)

    def test_prefix_called_once_coda_twice_and_hook_cleanup(self):
        m = optimized(dual(kl=0))
        stem = m.model.layers[0]; coda = m.model.layers[-1]
        with patch.object(stem, 'forward', wraps=stem.forward) as s, \
             patch.object(coda, 'forward', wraps=coda.forward) as c:
            m(**batch()).loss.backward()
            self.assertEqual((s.call_count, c.call_count), (1, 2))
        self.assertFalse(m.routed_steps[1]._forward_hooks)
        with patch.object(coda, 'forward', side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                m(**batch())
        self.assertFalse(m.routed_steps[1]._forward_hooks)

    def test_old_defaults_and_eval_unchanged(self):
        a = dual(kl=0)
        self.assertFalse(a.exit_options.shared_prefix)
        self.assertEqual(a.exit_options.ce_backend, 'chunked')
        b = optimized(copy.deepcopy(a)); a.eval(); b.eval()
        for policy in ('none', 'fixed'):
            a.stop_controller.policy = b.stop_controller.policy = policy
            a.stop_controller.fixed_depth = b.stop_controller.fixed_depth = 2
            torch.testing.assert_close(a(**batch()).logits, b(**batch()).logits, atol=0, rtol=0)
        with self.assertRaises(ValueError):
            optimized(dual(kl=.5))


if __name__ == '__main__':
    unittest.main()
