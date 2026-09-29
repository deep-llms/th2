"""Recomputation toggles must preserve gradients and optimizer continuation."""
import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
import torch
from tests.test_deep_kv import model, context, objective
from scripts.benchmark_capacity import boundary, inspect_attempt


class CheckpointCapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_loss_gradient_and_adam_continuation_all_arms(self):
        def setup():
            m=model(arm,checkpoint=True).train()
            o=torch.optim.AdamW(m.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1)
            s=torch.optim.lr_scheduler.LambdaLR(o,lambda i:1-.01*i)
            return m,o,s
        def step(m,o,s):
            o.zero_grad(set_to_none=True)
            loss=objective(m,m(context()));loss.backward()
            gradients={n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None}
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
            o.step();s.step()
            return loss.detach(),gradients
        for arm in 'ABCDEFG':
            m,o,s=setup()
            if m.aux is not None:
                with torch.no_grad():m.aux.out.weight.fill_(.01)
            for _ in range(2):step(m,o,s)
            saved=copy.deepcopy((m.state_dict(),o.state_dict(),s.state_dict()))
            loss,grads=step(m,o,s)
            for layers,lm,aux in itertools.product((True,False),repeat=3):
                with self.subTest(arm=arm,layers=layers,lm=lm,aux=aux):
                    other,opt,sched=setup()
                    other.load_state_dict(saved[0]);opt.load_state_dict(copy.deepcopy(saved[1]));sched.load_state_dict(saved[2])
                    other.checkpoint_layers=layers;other.checkpoint_lm=lm;other.checkpoint_aux=aux
                    actual,gradients=step(other,opt,sched)
                    torch.testing.assert_close(actual,loss,atol=1e-7,rtol=1e-6)
                    torch.testing.assert_close(gradients,grads,atol=1e-7,rtol=1e-5)
                    torch.testing.assert_close(other.state_dict(),m.state_dict(),atol=1e-7,rtol=1e-5)
                    torch.testing.assert_close(opt.state_dict(),o.state_dict(),atol=1e-7,rtol=1e-5)
                    self.assertEqual(sched.state_dict(),s.state_dict())

    def test_boundary_and_cap(self):
        tried=[]
        def probe(n):tried.append(n);return n<=27
        self.assertEqual(boundary(probe),dict(maximum_tested=27,first_failing=28,bounded=False))
        self.assertIn(27,tried);self.assertIn(28,tried)
        self.assertEqual(boundary(lambda n:True),dict(maximum_tested=64,first_failing=None,bounded=True))
        with self.assertRaises(ValueError):boundary(lambda n:False)

    def test_only_typed_oom_receipt_can_continue(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            with self.assertRaises(RuntimeError):inspect_attempt(path,1,'test')
            receipt=path/'capacity-oom-rank0.json'
            receipt.write_text(json.dumps(dict(status='cuda_oom',rank=0,variant='test')))
            self.assertEqual(inspect_attempt(path,1,'test')['status'],'cuda_oom')
            with self.assertRaises(ValueError):inspect_attempt(path,0,'test')
            with self.assertRaises(ValueError):inspect_attempt(path,1,'other')


if __name__=='__main__':unittest.main()
