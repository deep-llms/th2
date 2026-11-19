import copy
import unittest
import torch
from efficient_llm.warm_order.model import Config, Model as Dense, ORDERS
from efficient_llm.warm_order.frozen_router import Model


class FrozenRouterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2041)
        self.c = Config(d=16, heads=2, kv_heads=1, head_dim=8, ff=32,
                        vocab=37, length=12, prefix=3)
        self.x = torch.randint(37, (2, 12))
        self.parent = Dense(self.c)

    def model(self, arm):
        m = Model(self.c, arm, router_width=8)
        m.load_dense_backbone(self.parent.state_dict())
        return m

    def test_initial_parent_identity_and_all_arms_same_soft(self):
        values = []
        for arm in ('G', 'L', 'M'):
            m = self.model(arm).eval()
            torch.testing.assert_close(m.nll(self.x, True), self.parent.nll(self.x, ORDERS[0]), rtol=0, atol=0)
            values.append(m.nll(self.x))
        for value in values[1:]:
            torch.testing.assert_close(value, values[0], rtol=0, atol=0)

    def test_only_router_updates_and_earlier_gates_receive_gradient(self):
        for arm in ('G', 'L', 'M'):
            m = self.model(arm); frozen = m.frozen_state()
            before = m.depth_logits.detach().clone()
            opt = torch.optim.AdamW(m.router_parameters(), lr=.01, weight_decay=0)
            for _ in range(3):
                opt.zero_grad(set_to_none=True); m.nll(self.x).mean().backward()
                self.assertTrue(torch.isfinite(m.depth_logits.grad).all())
                self.assertTrue((m.depth_logits.grad.abs().sum(-1) > 0).all())
                opt.step(); m.assert_frozen(frozen)
            self.assertFalse(torch.equal(before, m.depth_logits))
            if arm == 'M':
                self.assertGreater(sum(float(h[1].weight.grad.abs().sum()) for h in m.depth_heads), 0)

    def test_causality_nontrivial_heads_and_independent_mixture(self):
        for arm in ('G', 'L', 'M'):
            m = self.model(arm).eval()
            with torch.no_grad():
                m.depth_logits.zero_()
                if arm == 'L': m.route_linear.weight.normal_(std=.2)
                if arm == 'M':
                    for head in m.depth_heads: head[-1].weight.normal_(std=.2)
            y = self.x.clone(); y[:, 7:] = (y[:, 7:] + 1) % 37
            for hard in (False, True):
                h, ps = m.hidden_and_gates(self.x, hard)
                hh, qs = m.hidden_and_gates(y, hard)
                torch.testing.assert_close(h[:, :7], hh[:, :7], rtol=0, atol=0)
                for a, b in zip(ps, qs): torch.testing.assert_close(a[:, :7], b[:, :7], rtol=0, atol=0)
            pos = torch.arange(12).expand_as(self.x); seg = torch.zeros_like(self.x)
            ref = m.stem(m.embed(self.x), pos, seg)
            for depth in range(4):
                p = m.gates(ref, depth)
                ref = sum(p[..., k, None] * block(ref, pos, seg) for k, block in enumerate(m.middle))
            ref = m.norm(m.coda(ref, pos, seg))
            torch.testing.assert_close(m.hidden(self.x), ref, rtol=2e-6, atol=2e-6)

    def test_optimizer_state_roundtrip_exact_all_arms(self):
        for arm in ('G', 'L', 'M'):
            m = self.model(arm); o = torch.optim.AdamW(m.router_parameters(), lr=.003)
            def step(model, opt):
                opt.zero_grad(set_to_none=True); model.nll(self.x).mean().backward(); opt.step()
            step(m,o)
            state = copy.deepcopy(m.state_dict()); optstate = copy.deepcopy(o.state_dict())
            step(m,o)
            n = self.model(arm); n.load_state_dict(state)
            oo = torch.optim.AdamW(n.router_parameters(), lr=.003); oo.load_state_dict(optstate); step(n,oo)
            for key, value in m.state_dict().items(): torch.testing.assert_close(value,n.state_dict()[key],rtol=0,atol=0)

    def test_reject_bad_backbone_and_detect_frozen_mutation(self):
        m = self.model('M'); before=m.frozen_state()
        state=dict(self.parent.state_dict()); state.pop(next(iter(state)))
        with self.assertRaises(ValueError): m.load_dense_backbone(state)
        with torch.no_grad():m.embed.weight[0,0].add_(1)
        with self.assertRaises(ValueError):m.assert_frozen(before)


if __name__ == '__main__': unittest.main()
