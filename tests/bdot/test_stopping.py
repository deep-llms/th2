"""STOP against a dense held-state reference, cache, gradients and persistence."""
import copy
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_model import model, batch
from bdot.model import BDOTForCausalLM
from bdot.stopping import StopController


def stopped(partial=False, probability=.05):
    options = dict(curriculum='none', step_film=False)
    if partial:
        options.update(route_scope='mlp', routed_mlp_fraction=.25, default_prior=0.)
    else:
        options.update(routing='fixed')
    m = model(**options)
    m.config.bdot_stop = dict(train_random_prob=probability, min_steps=2)
    m.stop_controller = StopController(m.config, m.options)
    return m


def dense_reference(m, depths):
    """Compute ALL Q/O/MLPs, then hold states. Independent sparse-path oracle."""
    ref = copy.deepcopy(m)
    del ref.stop_controller
    del ref.config.bdot_stop
    for layer in ref.routed_steps:
        original = layer._forward_full
        step = layer.step
        def forward(h, original=original, step=step, **kw):
            candidate = original(h, **kw)
            return torch.where((depths > step)[..., None], candidate, h)
        layer.forward = forward
    return ref


class StopTests(unittest.TestCase):
    def test_none_is_exact_original_including_gradients(self):
        for partial in (False, True):
            a = stopped(partial, 0.)
            b = copy.deepcopy(a)
            del b.stop_controller
            del b.config.bdot_stop
            x, y = a(**batch()), b(**batch())
            torch.testing.assert_close(x.logits, y.logits, atol=0, rtol=0)
            x.loss.backward(); y.loss.backward()
            for name, p in b.named_parameters():
                q = a.get_parameter(name)
                if p.grad is None:
                    self.assertIsNone(q.grad)
                else:
                    torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0, msg=name)

    def test_mixed_and_all_stopped_match_dense_held_state_oracle(self):
        for partial in (False, True):
            for depths in (torch.full((2,6), 2), torch.tensor([[2,3,4,2,4,3], [4,2,4,3,2,4]])):
                a = stopped(partial, 0.).eval()
                b = dense_reference(a, depths).eval()
                x = a(**batch(), bdot_stop_depths=depths); y = b(**batch())
                torch.testing.assert_close(x.logits, y.logits, atol=1e-6, rtol=1e-5)
                x.loss.backward(); y.loss.backward()
                for name, p in b.named_parameters():
                    q = a.get_parameter(name)
                    # Dense oracle computes zero-gradient projections that the
                    # sparse path never calls; None is the correct sparse result.
                    pg = torch.zeros_like(p) if p.grad is None else p.grad
                    qg = torch.zeros_like(q) if q.grad is None else q.grad
                    torch.testing.assert_close(pg, qg, atol=1e-6, rtol=1e-4, msg=name)
                self.assertTrue(torch.equal(a.stop_controller.last_depths, depths))

    def test_cached_isolated_forward_and_causality(self):
        for partial in (False, True):
            m = stopped(partial, 0.).eval()
            depths = torch.tensor([[2,3,4,2,4,3], [4,2,4,3,2,4]])
            inputs = batch()
            out = m(**inputs, bdot_stop_depths=depths).logits
            altered = copy.deepcopy(inputs); altered['input_ids'][0,2:] = 17
            other = m(**altered, bdot_stop_depths=depths).logits
            torch.testing.assert_close(out[0,:2], other[0,:2], atol=1e-6, rtol=1e-5)
            altered = copy.deepcopy(inputs); altered['input_ids'][0,:3] = 18
            other = m(**altered, bdot_stop_depths=depths).logits
            torch.testing.assert_close(out[0,3:], other[0,3:], atol=1e-6, rtol=1e-5)
            cache = None
            for t in range(6):
                r = m(input_ids=inputs['input_ids'][:,t:t+1],
                    position_ids=inputs['position_ids'][:,t:t+1],
                    attention_mask=inputs['attention_mask'][:,:,t:t+1,:t+1],
                    use_cache=True, past_key_values=cache, bdot_stop_depths=depths[:,t:t+1])
                cache = r.past_key_values
                torch.testing.assert_close(out[:,t], r.logits[:,0], atol=1e-6, rtol=1e-5)
                self.assertTrue(all(x.get_seq_length() == t+1 for x in cache.layers))

    def test_skipped_queries_and_both_mlp_parts_but_kv_still_written(self):
        m = stopped(True, 0.).eval(); m.stop_controller.policy = 'fixed'; m.stop_controller.fixed_depth = 2
        layer = m.routed_steps[2]; expert = m.model.bdot_experts[2]
        with patch.object(expert.q_proj, 'forward', wraps=expert.q_proj.forward) as q, \
             patch.object(expert.mlp, 'forward', wraps=expert.mlp.forward) as mlp, \
             patch.object(layer.core_mlp, 'forward', wraps=layer.core_mlp.forward) as core, \
             patch.object(layer.k_proj, 'forward', wraps=layer.k_proj.forward) as k:
            m(input_ids=torch.tensor([[1]]), use_cache=True)
            self.assertEqual((q.call_count, mlp.call_count, core.call_count, k.call_count), (0,0,0,1))

    def test_random_stop_checkpoint_replay_and_reload(self):
        for partial in (False, True):
            a = stopped(partial, .5)
            b = copy.deepcopy(a)
            b.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            x = a(**batch()); y = b(**batch())
            x.loss.backward(); y.loss.backward()
            torch.testing.assert_close(x.logits, y.logits, atol=0, rtol=0)
            self.assertTrue(torch.equal(a.stop_controller.last_depths, b.stop_controller.last_depths))
            self.assertTrue(torch.equal(a.load_counts, b.load_counts))
            for name, p in a.named_parameters():
                q = b.get_parameter(name)
                if p.grad is None:
                    self.assertIsNone(q.grad, name)
                else:
                    torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5, msg=name)
            a.finish_update()
            with tempfile.TemporaryDirectory() as tmp:
                a.save_pretrained(tmp)
                b = BDOTForCausalLM.from_pretrained(tmp, attn_implementation='sdpa').train()
            x = a(**batch()); y = b(**batch())
            torch.testing.assert_close(x.logits, y.logits, atol=0, rtol=0)
            self.assertTrue(torch.equal(a.stop_controller.last_depths, b.stop_controller.last_depths))

    def test_learned_initialization_and_guardrails(self):
        m = stopped(True, 0.).eval(); m.stop_controller.policy = 'learned'
        m(**batch())
        self.assertTrue(m.stop_controller.last_depths.eq(m.options.S).all())
        with torch.no_grad(): m.stop_controller.bias.fill_(10)
        m(**batch())
        self.assertTrue(m.stop_controller.last_depths.eq(2).all())
        with self.assertRaises(ValueError): m.train()(**batch())
        with self.assertRaises(ValueError):
            stopped().eval()(**batch(), bdot_stop_depths=torch.zeros(2,6,dtype=torch.long))


if __name__ == '__main__':
    unittest.main()
