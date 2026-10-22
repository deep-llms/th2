"""Semantic guards for dispatch fast paths, independent of a frozen source copy."""
import copy
import unittest

import torch
from test_model import model, batch


class DispatchTests(unittest.TestCase):
    def test_home_execution_with_learned_film_matches_generic_path(self):
        a = model(routing='fixed', curriculum='none')
        with torch.no_grad():
            for step in a.routed_steps:
                step.film_gain.normal_(0, .2)
                step.film_bias.normal_(0, .1)
        b = copy.deepcopy(a)
        for step in b.routed_steps:
            def generic(h, mask, positions, cache, kwargs, step=step):
                n = h.shape[0]*h.shape[1]
                return step.execute(h, torch.full((n, 1), step.step, dtype=torch.long),
                                    h.new_ones(n, 1), mask, positions, cache, kwargs)
            step.execute_home = generic
        x, y = a(**batch()), b(**batch())
        torch.testing.assert_close(x.logits, y.logits, rtol=0, atol=0)
        x.loss.backward()
        y.loss.backward()
        for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
            torch.testing.assert_close(pa.grad, pb.grad, atol=1e-7, rtol=1e-5, msg=name)

    def test_hard_execution_preserves_straight_through_gate_gradient(self):
        a = model(r=3)
        b = copy.deepcopy(a)
        sa, sb = a.routed_steps[1], b.routed_steps[1]
        inputs = batch()
        h = torch.randn(2, 6, 16, requires_grad=True)
        other = h.detach().clone().requires_grad_()
        choices = (torch.arange(12) % 4).reshape(-1, 1)
        scores = torch.randn(12, 1, requires_grad=True)
        rs = scores.detach().clone().requires_grad_()
        gates = (scores-scores.detach()).exp()
        rgates = (rs-rs.detach()).exp()
        positions = a.model.rotary_emb(h, inputs['position_ids'])
        x = sa.execute(h, choices, gates, inputs['attention_mask'], positions, None, {}, all_executed=True)
        y = sb.execute(other, choices, rgates, inputs['attention_mask'], positions, None, {})
        torch.testing.assert_close(x, y, rtol=0, atol=0)
        x.square().sum().backward()
        y.square().sum().backward()
        torch.testing.assert_close(scores.grad, rs.grad, rtol=0, atol=0)
        torch.testing.assert_close(h.grad, other.grad, rtol=0, atol=0)

    def test_metrics_snapshot_survives_next_forward_and_reads_are_pure(self):
        m = model(curriculum='none')
        m(**batch())
        m.finish_update()
        self.assertIsNone(m._last_update_metrics)
        expected = copy.deepcopy(m)
        values = expected.last_update_metrics
        m(**batch())
        state = {name: tensor.clone() for name, tensor in m.state_dict().items()}
        self.assertEqual(m.last_update_metrics, values)
        self.assertEqual(m.last_update_metrics, values)
        for name, tensor in m.state_dict().items():
            torch.testing.assert_close(tensor, state[name], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
