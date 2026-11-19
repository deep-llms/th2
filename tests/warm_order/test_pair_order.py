import copy
import unittest
import torch
from efficient_llm.warm_order.model import Config, Model as Dense, ORDERS
from efficient_llm.warm_order.pair_order import Model


class PairOrderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2044)
        self.c = Config(d=16, heads=2, kv_heads=1, head_dim=8, ff=32,
                        vocab=37, length=12, prefix=3)
        self.x = torch.randint(37, (2, 12))
        self.parent = Dense(self.c)

    def model(self, arm):
        m = Model(self.c, arm, router_width=8)
        m.load_dense(self.parent.state_dict())
        return m

    def test_control_identity_gradient_and_initial_hard(self):
        dense = copy.deepcopy(self.parent)
        dense.nll(self.x, ORDERS[0]).mean().backward()
        control = self.model('C')
        torch.testing.assert_close(control.nll(self.x), dense.nll(self.x, ORDERS[0]), rtol=0, atol=0)
        control.nll(self.x).mean().backward()
        for name, p in dense.named_parameters():
            torch.testing.assert_close(dict(control.named_parameters())[name].grad, p.grad, rtol=2e-5, atol=2e-7)
        for arm in ('G', 'M'):
            torch.testing.assert_close(self.model(arm).nll(self.x, True),
                                       dense.nll(self.x, ORDERS[0]), rtol=0, atol=0)
        torch.testing.assert_close(self.model('G').nll(self.x), self.model('M').nll(self.x), rtol=0, atol=0)

    def test_nontrivial_causal_gates_and_outputs(self):
        m = self.model('M').eval()
        with torch.no_grad():
            m.pair_logits.zero_()
            for head in m.pair_heads: head[-1].weight.normal_(std=.3)
        changed = self.x.clone(); changed[:, 7:] = (changed[:, 7:]+1) % 37
        for hard in (False, True):
            h, ps = m.hidden_and_gates(self.x, hard)
            other, qs = m.hidden_and_gates(changed, hard)
            torch.testing.assert_close(h[:, :7], other[:, :7], rtol=0, atol=0)
            for p, q in zip(ps, qs): torch.testing.assert_close(p[:, :7], q[:, :7], rtol=0, atol=0)

    def test_independent_reference_gradient_both_pairs(self):
        m = self.model('M').double(); ref = copy.deepcopy(m)
        with torch.no_grad():
            m.pair_logits.zero_(); ref.pair_logits.zero_()
        pos = torch.arange(12).expand_as(self.x); seg = torch.zeros_like(self.x)
        h = ref.stem(ref.embed(self.x), pos, seg)
        for pair in range(2):
            head = ref.pair_heads[pair]
            z = torch.nn.functional.layer_norm(h, (self.c.d,), head[0].weight, head[0].bias, head[0].eps)
            z = torch.nn.functional.silu(torch.nn.functional.linear(z, head[1].weight, head[1].bias))
            p = (ref.pair_logits[pair] + torch.nn.functional.linear(z, head[3].weight, head[3].bias)).softmax(-1)
            a, b = ref.middle[2*pair:2*pair+2]
            h = p[..., :1]*b(a(h,pos,seg),pos,seg) + p[..., 1:]*a(b(h,pos,seg),pos,seg)
        reference = ref.norm(ref.coda(h,pos,seg))
        actual = m.hidden(self.x)
        torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
        actual.square().mean().backward(); reference.square().mean().backward()
        for name, p in m.named_parameters():
            torch.testing.assert_close(p.grad, dict(ref.named_parameters())[name].grad, rtol=1e-9, atol=1e-10)
        self.assertTrue((m.pair_logits.grad.abs().sum(-1)>0).all())

    def test_optimizer_roundtrip_all_arms_and_tamper(self):
        for arm in ('C','G','M'):
            m = self.model(arm); opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=.003)
            def step(model, optimizer):
                optimizer.zero_grad(set_to_none=True); model.nll(self.x).mean().backward(); optimizer.step()
            step(m,opt)
            state, optim = copy.deepcopy(m.state_dict()), copy.deepcopy(opt.state_dict())
            step(m,opt)
            n = self.model(arm); n.load_state_dict(state)
            op = torch.optim.AdamW([p for p in n.parameters() if p.requires_grad], lr=.003)
            op.load_state_dict(optim); step(n,op)
            for name,v in m.state_dict().items(): torch.testing.assert_close(v,n.state_dict()[name],rtol=0,atol=0)
        before=copy.deepcopy(m.state_dict()); bad=dict(self.parent.state_dict()); bad['unknown']=torch.zeros(1)
        with self.assertRaises(ValueError): m.load_dense(bad)
        for name,v in before.items(): torch.testing.assert_close(v,m.state_dict()[name],rtol=0,atol=0)


if __name__ == '__main__': unittest.main()
