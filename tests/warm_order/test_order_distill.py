import copy
import unittest
import torch
from torch.nn import functional as F
from efficient_llm.warm_order.model import Config, Model
from efficient_llm.warm_order.order_distill import chunked_objective, objective, ARMS, SWAP


class DistillTests(unittest.TestCase):
    def test_chunked_exact_loss_gradients_and_teacher_detached(self):
        torch.manual_seed(8)
        h = torch.randn(2, 5, 4, requires_grad=True)
        w = torch.randn(17, 4, requires_grad=True)
        t = torch.randn(2, 5, 4, requires_grad=True)
        tw = torch.randn(17, 4, requires_grad=True)
        y = torch.randint(17, (2, 5))
        got = chunked_objective(h, w, t, tw, y, chunk=3)
        logp = F.log_softmax(F.linear(h, w), -1)
        logq = F.log_softmax(F.linear(t.detach(), tw.detach()), -1)
        ref = torch.stack((F.nll_loss(logp.reshape(-1,17), y.flatten()),
                           (logq.exp()*(logq-logp)).sum(-1).mean()))
        torch.testing.assert_close(got, ref)
        frozen_grads = torch.autograd.grad(got.sum(), (t,tw), allow_unused=True, retain_graph=True)
        self.assertEqual(frozen_grads, (None,None))
        gg = torch.autograd.grad(got[0]+.5*got[1], (h,w), retain_graph=True)
        gr = torch.autograd.grad(ref[0]+.5*ref[1], (h,w))
        for a,b in zip(gg,gr):torch.testing.assert_close(a,b)
        self.assertIsNone(t.grad);self.assertIsNone(tw.grad)

    def test_all_arm_gradients_teacher_frozen_and_causality(self):
        torch.set_num_threads(1);torch.manual_seed(4)
        cfg=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=37,length=12,prefix=3)
        teacher=Model(cfg).eval()
        for p in teacher.parameters():p.requires_grad_(False)
        saved={k:v.clone() for k,v in teacher.state_dict().items()}
        x=torch.randint(37,(2,12))
        for arm in ARMS:
            student=copy.deepcopy(teacher)
            for p in student.parameters():p.requires_grad_(True)
            student.train();loss,ce,kl=objective(student,teacher,x,arm,chunk=4)
            loss.backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters()))
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))
            if arm in ('C2','K2'):self.assertLess(abs(float(kl.detach())),1e-6)
            changed=x.clone();changed[:,6:]=(changed[:,6:]+1)%37
            torch.testing.assert_close(student.hidden(x,SWAP)[:,:6],student.hidden(changed,SWAP)[:,:6],rtol=0,atol=0)
            for k,v in teacher.state_dict().items():self.assertTrue(torch.equal(saved[k],v))
        with self.assertRaises(ValueError):objective(Model(cfg),Model(cfg),x,'PK2')

if __name__=='__main__':unittest.main()
