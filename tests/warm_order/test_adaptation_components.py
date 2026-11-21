import unittest

import torch
from torch.nn import functional as F

from efficient_llm.warm_order.model import Model, Config
from efficient_llm.warm_order.adaptation_analysis import CANONICAL, SWAPPED, native_trace
from efficient_llm.warm_order.adaptation_components import component_crossovers, score_crossovers, ATOL, RTOL


class ComponentCrossovers(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(122)
        self.m = Model(Config(d=16, heads=4, kv_heads=2, head_dim=4,
                              ff=24, vocab=31, prefix=3, length=12)).eval().requires_grad_(False)
        self.x = torch.randint(0, 31, (2, 12))

    def test_native_component_and_same_path_all_cells(self):
        for kind in ('attention', 'ffn'):
            result = component_crossovers(self.m, self.x, 2, kind, donor_path=SWAPPED)
            for cell in result['cells'].values():
                torch.testing.assert_close(cell['hidden'], self.m.hidden(self.x, SWAPPED), atol=ATOL, rtol=RTOL)
                logits = F.linear(cell['hidden'][:, 2:-1], self.m.embed.weight)
                nll = F.cross_entropy(logits.flatten(0, 1), self.x[:, 3:].flatten(), reduction='none').reshape(2, 9).mean(1)
                torch.testing.assert_close(nll, self.m.nll(self.x, SWAPPED), atol=ATOL, rtol=RTOL)
            if kind=='attention':
                for p in result['parts']:
                    self.assertEqual(p['pattern'].triu(1).count_nonzero(), 0)
                    torch.testing.assert_close(p['pattern'].sum(-1), torch.ones_like(p['pattern'].sum(-1)))

    def test_crossed_cells_match_explicit_recipient_suffix(self):
        _, records = native_trace(self.m, self.x, SWAPPED)
        before = next(r['input'] for r in records if r['block']=='middle.2')
        pos = torch.arange(12).expand_as(self.x); seg = torch.zeros_like(self.x)
        for kind in ('attention', 'ffn'):
            result = component_crossovers(self.m, self.x, 2, kind)
            # Confirms real hybrid differences, not four copies of native output.
            self.assertGreater(float((result['cells']['10']['update']-result['cells']['00']['update']).abs().max()), 0)
            for cell in result['cells'].values():
                block = self.m.middle[2]
                if kind=='attention':
                    h = before+cell['update']; h = h+block.mlp(block.post_attention_layernorm(h))
                else:
                    h = block.attention(before, pos, seg)+cell['update']
                for index in (1, 3): h = self.m.middle[index](h, pos, seg)
                expected = self.m.norm(self.m.coda(h, pos, seg))
                torch.testing.assert_close(cell['hidden'], expected, rtol=0, atol=0)
            # The whole donor state was never copied into the recipient graph.
            self.assertGreater(float((result['cells']['11']['hidden']-result['donor_hidden']).abs().max()), 0)
            scored = score_crossovers(self.m, self.x, result)
            for key, cell in result['cells'].items():
                logits = F.linear(cell['hidden'][:, 2:-1], self.m.embed.weight)
                nll = F.cross_entropy(logits.flatten(0, 1), self.x[:, 3:].flatten(), reduction='none').reshape(2, 9)
                torch.testing.assert_close(scored['cells'][key]['nll_sum'], nll.double().sum(1), rtol=0, atol=1e-6)
                torch.testing.assert_close(scored['cells'][key]['token_count'], torch.full((2,), 9, dtype=torch.int64), rtol=0, atol=0)
            effect = scored['effects_nll_sum']
            torch.testing.assert_close(effect['first_with_recipient_second']+effect['second_with_donor_first'], effect['joint'], rtol=0, atol=1e-12)

    def test_future_causality_both_directions_and_no_mutation(self):
        changed = self.x.clone(); changed[:, 7:] = (changed[:, 7:]+1)%31
        state = {k: v.clone() for k, v in self.m.state_dict().items()}
        rng = torch.get_rng_state().clone()
        for donor, recipient in ((CANONICAL, SWAPPED), (SWAPPED, CANONICAL)):
            for kind in ('attention', 'ffn'):
                a = component_crossovers(self.m, self.x, 1, kind, donor_path=donor, recipient_path=recipient)
                b = component_crossovers(self.m, changed, 1, kind, donor_path=donor, recipient_path=recipient)
                for key in a['cells']:
                    torch.testing.assert_close(a['cells'][key]['hidden'][:, :7], b['cells'][key]['hidden'][:, :7], rtol=0, atol=0)
        for key, value in self.m.state_dict().items(): torch.testing.assert_close(value, state[key], rtol=0, atol=0)
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.m.modules()))

    def test_invalid_component_precision_and_cleanup_on_failure(self):
        for target, kind in [(True, 'ffn'), (4, 'ffn'), (1, 'other')]:
            with self.assertRaises(ValueError): component_crossovers(self.m, self.x, target, kind)
        def fail(*args): raise RuntimeError('injected')
        # Fail on third coda call, inside the first intervention (after two references).
        count = []
        def fail_third(*args):
            count.append(1)
            if len(count)==3: fail()
        hook = self.m.coda.register_forward_hook(fail_third)
        try:
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                component_crossovers(self.m, self.x, 2, 'attention')
        finally: hook.remove()
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.m.modules()))
        self.m.double()
        with self.assertRaisesRegex(ValueError, 'FP32'): component_crossovers(self.m, self.x, 2, 'ffn')


if __name__ == '__main__': unittest.main()
