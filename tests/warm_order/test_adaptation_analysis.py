import copy
import unittest

import torch
from torch.nn import functional as F

from efficient_llm.warm_order.model import Model, Config
from efficient_llm.warm_order.adaptation_analysis import (
    CANONICAL, SWAPPED, native_trace, weight_cells, loss_contrasts,
)


class AdaptationAnalysis(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(71021)
        self.model = Model(Config(d=16, heads=2, kv_heads=1, head_dim=8,
                                  ff=32, vocab=37, prefix=3, length=12)).eval().requires_grad_(False)
        self.x = torch.randint(0, 37, (2, 12))

    def test_native_capture_parity_and_actual_branch_outputs(self):
        rng = torch.get_rng_state().clone()
        state = {k: v.clone() for k, v in self.model.state_dict().items()}
        for path in (CANONICAL, SWAPPED):
            hidden, trace = native_trace(self.model, self.x, path)
            torch.testing.assert_close(hidden, self.model.hidden(self.x, path), rtol=0, atol=0)
            self.assertEqual([r['block'] for r in trace[1:5]], [f'middle.{i}' for i in path])
            for record in trace:
                block = self.model.get_submodule(record['block'])
                expected = block.mlp(record['ffn_input_normalized'])
                torch.testing.assert_close(record['ffn_update'], expected, rtol=0, atol=0)
                torch.testing.assert_close(record['post_attention']+expected, record['output'], rtol=0, atol=0)
            logits = F.linear(hidden[:, 2:-1], self.model.embed.weight)
            nll = F.cross_entropy(logits.flatten(0, 1), self.x[:, 3:].flatten(), reduction='none').reshape(2, 9).mean(1)
            torch.testing.assert_close(nll, self.model.nll(self.x, path), rtol=0, atol=0)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.model.modules()))

    def test_causality_and_cleanup_on_failed_forward(self):
        changed = self.x.clone(); changed[:, 7:] = (changed[:, 7:]+1) % 37
        h, a = native_trace(self.model, self.x, SWAPPED)
        g, b = native_trace(self.model, changed, SWAPPED)
        torch.testing.assert_close(h[:, :7], g[:, :7], rtol=0, atol=0)
        for r, s in zip(a, b):
            torch.testing.assert_close(r['attention_update'][:, :7], s['attention_update'][:, :7], rtol=0, atol=0)
        def fail(*args): raise RuntimeError('injected failure')
        handle = self.model.coda.register_forward_hook(fail)
        try:
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                native_trace(self.model, self.x, CANONICAL)
        finally:
            handle.remove()
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.model.modules()))

    def test_weight_cells_exact_partition_endpoints_and_no_alias(self):
        adapted = copy.deepcopy(self.model)
        with torch.no_grad():
            for i, p in enumerate(adapted.parameters()): p.add_((i+1)*.0001)
        cells = dict((name, (model, sources)) for name, model, sources in weight_cells(self.model, adapted))
        for name, (model, sources) in cells.items():
            for key, value in model.state_dict().items():
                bit = int(name[0] if key.startswith(('middle.1.', 'middle.2.')) else name[1])
                self.assertEqual(sources[key], bit)
                reference = (adapted if bit else self.model).state_dict()[key]
                torch.testing.assert_close(value, reference, rtol=0, atol=0)
                self.assertNotEqual(value.data_ptr(), reference.data_ptr())
            self.assertEqual(sources['embed.weight'], int(name[1]))
        for name, original in [('00', self.model), ('11', adapted)]:
            torch.testing.assert_close(cells[name][0].nll(self.x, SWAPPED), original.nll(self.x, SWAPPED), rtol=0, atol=0)
        bad = copy.deepcopy(adapted); bad.embed.weight[0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'): list(weight_cells(self.model, bad))

    def test_joint_only_recovery_and_rejection(self):
        # No individual rescue, but a positive joint effect; no additive attribution.
        values = {k: torch.tensor([v, 2*v]) for k, v in [('00', 5.), ('10', 5.), ('01', 5.), ('11', 3.)]}
        out = loss_contrasts(values)
        torch.testing.assert_close(out['pair_parent_context'], torch.zeros(2, dtype=torch.float64))
        torch.testing.assert_close(out['interaction'], torch.tensor([2., 4.], dtype=torch.float64))
        self.model.train()
        with self.assertRaises(ValueError): native_trace(self.model, self.x, CANONICAL)
        self.model.eval()
        with self.assertRaises(ValueError): native_trace(self.model, self.x, (0, 1, 1, 3))
        with self.assertRaises(ValueError): loss_contrasts({'00': torch.tensor(1.)})


if __name__ == '__main__':
    unittest.main()
