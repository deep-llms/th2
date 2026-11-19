import unittest
import torch
from efficient_llm.warm_order.soft_exploration import epsilon_at, floor_probabilities


class ExplorationTests(unittest.TestCase):
    def test_schedule_and_boundaries(self):
        expected = {0: 0., 250: .2, 500: .4, 750: .4, 1000: .4,
                    1250: .2, 1500: 0., 1999: 0.}
        for update, value in expected.items():
            self.assertAlmostEqual(epsilon_at(update), value)
        for update in (-1, True, 1.5):
            with self.assertRaises(ValueError): epsilon_at(update)

    def test_floor_normalization_and_argmax(self):
        torch.manual_seed(1)
        p = torch.randn(3, 8, 4, dtype=torch.double).softmax(-1)
        for epsilon in (0., .1, .4):
            q = floor_probabilities(p, epsilon)
            torch.testing.assert_close(q.sum(-1), torch.ones_like(q[..., 0]))
            self.assertTrue(bool((q >= epsilon / 4).all()))
            self.assertTrue(torch.equal(p.argmax(-1), q.argmax(-1)))
        self.assertTrue(torch.equal(p, floor_probabilities(p, 0.)))

    def test_gradients_through_logits_and_branches(self):
        torch.manual_seed(2)
        logits = torch.randn(1, 2, 4, dtype=torch.double, requires_grad=True)
        branches = torch.randn(1, 2, 4, 3, dtype=torch.double, requires_grad=True)
        def f(z, b):
            return (b * floor_probabilities(z.softmax(-1), .4).unsqueeze(-1)).sum(-2)
        self.assertTrue(torch.autograd.gradcheck(f, (logits, branches)))

    def test_model_floor_causality_and_eval_disable(self):
        from efficient_llm.warm_order.soft_exploration import Model
        from efficient_llm.warm_order.model import Config
        torch.manual_seed(3)
        m=Model(Config(d=8,heads=1,kv_heads=1,head_dim=8,ff=16,vocab=19,length=8,prefix=2))
        x=torch.randint(19,(2,8));m.train();m.set_training_update(500)
        h,ps=m.hidden_and_gates(x)
        self.assertTrue(all(bool((p>=.1).all()) for p in ps))
        y=x.clone();y[:,4:]=(y[:,4:]+1)%19
        hh,qs=m.hidden_and_gates(y)
        torch.testing.assert_close(h[:,:4],hh[:,:4],rtol=0,atol=0)
        for p,q in zip(ps,qs):torch.testing.assert_close(p[:,:4],q[:,:4],rtol=0,atol=0)
        restored=Model(m.c);restored.load_state_dict(m.state_dict());restored.set_training_update(500)
        torch.testing.assert_close(h,restored.hidden(x),rtol=0,atol=0)
        m.eval();_,p=m.hidden_and_gates(x)
        m.disable_exploration();_,q=m.hidden_and_gates(x)
        for a,b in zip(p,q):torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertTrue(bool((p[0].max(-1).values>.99).all()))


if __name__ == '__main__': unittest.main()
