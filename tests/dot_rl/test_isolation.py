"""Independent adversarial checks on the same submitted model/RL code."""
import copy,unittest
import numpy as np
import torch
from torch.nn import functional as F
from efficient_llm.dot_pool.rl import Library,Policy,batch,tensor_hash
from efficient_llm.dot_pool.rl_run import optim,update
from efficient_llm.dot_pool.train import seed
from tests.dot_rl.test_core import tiny,rows

class Isolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_answer_and_all_future_tokens(self):
        seed(18);m=Library(tiny());p=Policy(16)
        with torch.no_grad():p.W.normal_(std=.5)
        x,pos,_=batch(rows(),torch.device('cpu'));x=torch.cat([x,torch.randint(2240,(8,3))],1)
        z=x.clone();mask=torch.arange(x.shape[1])[None,:]>pos[:,None];z[mask]=torch.randint(2240,(int(mask.sum()),))
        seed(33);a=m(x,pos,p);seed(33);b=m(z,pos,p)
        self.assertTrue(torch.equal(a['paths'],b['paths']));torch.testing.assert_close(a['logits'],b['logits'],rtol=0,atol=0)
    def test_grouped_gradients_match_per_question(self):
        seed(28);m=Library(tiny());reference=copy.deepcopy(m);x,pos,y=batch(rows(),torch.device('cpu'));paths=torch.randint(4,(8,8))
        F.cross_entropy(m(x,pos,paths=paths)['logits'],y).backward()
        for i in range(8):
            (F.cross_entropy(reference(x[i:i+1],pos[i:i+1],paths=paths[i:i+1])['logits'],y[i:i+1])/8).backward()
        for a,b in zip(m.parameters(),reference.parameters()):torch.testing.assert_close(a.grad,b.grad,atol=1e-6,rtol=1e-4)
    def test_both_passes_use_same_weights_and_frozen_FR(self):
        seed(38);m=Library(tiny());p=Policy(16);device=torch.device('cpu');opt,po=optim(m,p,device);before=tensor_hash(m);seen=[]
        handle=m.register_forward_pre_hook(lambda module,args:seen.append(tensor_hash(module)))
        update(m,p,opt,po,rows(),rows(),device,1,active=True)
        handle.remove();self.assertEqual(seen,[before,before]);self.assertNotEqual(before,tensor_hash(m))
        m.requires_grad_(False);before=tensor_hash(m);_,po=optim(m,p,device,True)
        for i in range(3):update(m,p,None,po,None,rows(),device,i+1,active=True,group=8)
        self.assertEqual(before,tensor_hash(m))
    def test_fixed_policy_has_no_question_signal(self):
        p=Policy(16,fixed=True)
        self.assertFalse(p.W.requires_grad)
        with torch.no_grad():p.E.normal_();p.b.normal_()
        a=p(torch.randn(20,16),3,torch.ones(20,dtype=torch.long));b=p(torch.randn(20,16)*100,3,torch.ones(20,dtype=torch.long))
        torch.testing.assert_close(a,b,rtol=0,atol=0)
