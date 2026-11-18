import copy, unittest
import numpy as np
import torch
from efficient_llm.warm_order.model import Config
from efficient_llm.warm_order.full_context_order import Model as Full
from efficient_llm.warm_order.balance_units import Model

class BalanceTests(unittest.TestCase):
 def test_token_units_partial_windows_and_unchanged_forward_gradient(self):
  torch.set_num_threads(1);torch.manual_seed(91);c=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=37,length=12,prefix=3)
  for mode in ('H','S'):
   old=Full(c,mode,5,2,0);x=torch.randint(37,(2,12))
   for scheme in ('T','O'):
    new=Model(c,mode+scheme,5,2,0);new.load_state_dict(old.state_dict());old.zero_grad();new.zero_grad();old.loads.zero_();new.loads.zero_()
    torch.manual_seed(5);a=old.nll(x,(0,1,2,3)).sum();a.backward();torch.manual_seed(5);b=new.nll(x,(0,1,2,3)).sum();b.backward();torch.testing.assert_close(a,b,rtol=0,atol=0)
    for (n,p),(m,q) in zip(old.named_parameters(),new.named_parameters()):
     self.assertEqual(n,m)
     if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
    if scheme=='T':torch.testing.assert_close(new.loads,torch.full((4,),24.),rtol=0,atol=0)
    new.finish_update();self.assertEqual(new.balance.count_nonzero(),0);self.assertEqual(new.loads.count_nonzero(),0)
 def test_frozen_canonical_controller_drift_and_token_invariance(self):
  p=np.exp(2*np.eye(4));p/=p.sum(-1,keepdims=True);found={}
  for scheme in ('window','token','off'):
   bias=np.zeros(4);first=None
   for step in range(1,2001):
    choice=(p+bias).argmax(-1)
    if not np.array_equal(choice,np.arange(4)) and first is None:first=step
    loads=np.bincount(choice,weights=[4,5,4,5] if scheme=='window' else [512]*4,minlength=4)
    if scheme!='off':bias+=.001*np.sign(loads.mean()-loads)
   found[scheme]=first
  self.assertEqual(found,dict(window=309,token=None,off=None))
if __name__=='__main__':unittest.main()
