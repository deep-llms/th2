import unittest
import torch
from efficient_llm.warm_order.model import Model as Dense, Config, ORDERS
from efficient_llm.warm_order.soft_route import Model, combine


class SoftRouteTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(91)
        self.c = Config(d=16, heads=2, kv_heads=1, head_dim=8, ff=32, vocab=37, length=12, prefix=3)
        self.x = torch.randint(37, (2, 12))

    def test_mixture_gradients_and_projection(self):
        output = torch.randn(1, 3, 4, 5, dtype=torch.double, requires_grad=True)
        logits = torch.randn(1, 3, 4, dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda a,b: combine(a,b.softmax(-1)), (output, logits)))
        p = logits.softmax(-1)
        expected = torch.stack([output[0,t,int(p[0,t].argmax())] for t in range(3)])[None]
        torch.testing.assert_close(combine(output,p,True),expected,rtol=0,atol=0)

    def test_parent_identity_and_initial_perturbation(self):
        dense = Dense(self.c).eval()
        for mode in ('dense', 'global', 'dynamic'):
            m = Model(self.c, mode).eval(); m.load_dense(dense.state_dict())
            torch.testing.assert_close(m.nll(self.x,True),dense.nll(self.x,ORDERS[0]),rtol=0,atol=0)
            self.assertLess(float((m.nll(self.x)-dense.nll(self.x,ORDERS[0])).abs().max().detach()), .001)

    def test_dynamic_causality_and_reference(self):
        m = Model(self.c).eval()
        with torch.no_grad():
            m.depth_logits.zero_(); m.route_linear.weight.normal_(std=.1)
        changed = self.x.clone(); changed[:,7:] = (changed[:,7:]+1)%37
        for hard in (False, True):
            h, p = m.hidden_and_gates(self.x,hard)
            hh, pp = m.hidden_and_gates(changed,hard)
            torch.testing.assert_close(h[:,:7],hh[:,:7],rtol=0,atol=0)
            for a,b in zip(p,pp):
                torch.testing.assert_close(a[:,:7],b[:,:7],rtol=0,atol=0)
                torch.testing.assert_close(a.sum(-1),torch.ones_like(a[...,0]))
            pos=torch.arange(12).expand_as(self.x); seg=torch.zeros_like(self.x)
            ref=m.stem(m.embed(self.x),pos,seg)
            for depth in range(4):
                gates=m.gates(ref,depth); branches=[block(ref,pos,seg) for block in m.middle]
                if hard:
                    ref=torch.stack([torch.stack([branches[int(gates[b,t].argmax())][b,t] for t in range(12)]) for b in range(2)])
                else:
                    ref=sum(gates[...,k,None]*branches[k] for k in range(4))
            ref=m.norm(m.coda(ref,pos,seg))
            torch.testing.assert_close(h,ref,atol=2e-6,rtol=2e-6)

    def test_body_and_router_gradients(self):
        m=Model(self.c)
        m.nll(self.x).mean().backward()
        for p in (m.depth_logits,m.route_linear.weight):
            self.assertTrue(torch.isfinite(p.grad).all()); self.assertGreater(float(p.grad.abs().sum()),0)
        for block in m.middle:
            grads=[p.grad for p in block.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads),0)

if __name__=='__main__': unittest.main()
