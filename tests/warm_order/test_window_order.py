import copy,unittest
import torch
from efficient_llm.warm_order.model import Config
from efficient_llm.warm_order.window_order import Model,bounds,prior_features

class WindowTests(unittest.TestCase):
 def test_grouped_windows_reference_gradients_partial_and_batch(self):
  torch.set_num_threads(1);torch.manual_seed(1);c=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=37,length=13,prefix=3)
  for arm in ('WF','WR','WG','WL'):
   m=Model(c,arm,window=5,stride=2).eval();ref=copy.deepcopy(m);h=torch.randn(3,13,16,requires_grad=True);hr=h.detach().clone().requires_grad_();x=torch.randint(37,(3,13));got,ch=m.layer(h,x,1);spans=bounds(13,5,2);choices,gates=ref.choose(hr,x,spans,1);parts=[]
   for j,(a,b) in enumerate(spans):
    per=[]
    for i in range(3):
     q=hr[i:i+1,a:b];pos=torch.arange(a,b)[None];seg=torch.zeros_like(pos);v=ref.middle[int(choices[i,j])](q,pos,seg);per.append(q+gates[i,j]*(v-q))
    parts.append(torch.cat(per))
   want=torch.cat(parts,1);torch.testing.assert_close(got,want,rtol=1e-5,atol=1e-6);got.square().mean().backward();want.square().mean().backward();torch.testing.assert_close(h.grad,hr.grad,rtol=2e-5,atol=2e-6)
   for (n,p),(nn,pp) in zip(m.named_parameters(),ref.named_parameters()):
    if p.grad is not None:torch.testing.assert_close(p.grad,pp.grad,rtol=2e-4,atol=2e-6)
   torch.testing.assert_close(m.hidden(x),torch.cat([m.hidden(z[None]) for z in x]),rtol=2e-5,atol=2e-6)

 def test_exclusive_prior_does_not_use_current_window_even_numerically(self):
  h=torch.randn(2,13,16);spans=bounds(13,5,2);a=prior_features(h,spans)
  z=h.clone();z[:,3:8]*=1e7;b=prior_features(z,spans);torch.testing.assert_close(a[:,:2],b[:,:2],rtol=0,atol=0)

 def test_future_independence_and_router_learning_balance(self):
  torch.set_num_threads(1);torch.manual_seed(4);c=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=37,length=13,prefix=3)
  for arm in ('WF','WR','WG','WL'):
   m=Model(c,arm,5,2).eval()
   with torch.no_grad():m.router.weight.normal_(std=.2);m.router.bias.normal_(std=.1)
   x=torch.randint(37,(2,13));a=m.hidden(x)
   for cut in (1,3,5,8,12):
    z=x.clone();z[:,cut:]=(z[:,cut:]+7)%37;b=m.hidden(z);torch.testing.assert_close(a[:,:cut],b[:,:cut],rtol=0,atol=0)
  m=Model(c,'WL',5,2).train();loss=m.nll(x,(0,1,2,3)).mean();loss.backward();self.assertGreater(float(m.router.weight.grad.abs().sum()),0);before=m.balance.clone();self.assertGreater(float(m.loads.sum()),0);m.finish_update();self.assertEqual(float(m.loads.sum()),0);self.assertFalse(torch.equal(before,m.balance))
if __name__=='__main__':unittest.main()
