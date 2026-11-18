import copy,unittest
import torch
from efficient_llm.warm_order.model import Config
from efficient_llm.warm_order.window_order import Model as Parent,bounds
from efficient_llm.warm_order.window_film import Model

class FilmTests(unittest.TestCase):
 def test_identity_and_reference_gradients_and_causality(self):
  torch.set_num_threads(1);torch.manual_seed(19);c=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=37,length=13,prefix=3);x=torch.randint(37,(3,13))
  for arm in ('WG','WL'):
   parent=Parent(c,arm,5,2).eval()
   with torch.no_grad():parent.router.weight.normal_(std=.1);parent.router.bias.normal_(std=.1);parent.balance.uniform_(-.01,.01)
   for film in (False,True):
    m=Model(c,arm,5,2,film=film).eval();m.load_parent(parent.state_dict());torch.testing.assert_close(m.hidden(x),parent.hidden(x),rtol=0,atol=0)
    if film:
     with torch.no_grad():m.film_gamma.normal_(std=.1);m.film_beta.normal_(std=.1)
    ref=copy.deepcopy(m);h=torch.randn(3,13,16,requires_grad=True);hr=h.detach().clone().requires_grad_();got,ch=m.layer(h,x,1);spans=bounds(13,5,2);choices,gates=ref.choose(hr,x,spans,1);parts=[]
    for j,(a,b) in enumerate(spans):
     per=[]
     for i in range(3):
      q=hr[i:i+1,a:b];z=q*(1+ref.film_gamma[1])+ref.film_beta[1];pos=torch.arange(a,b)[None];v=ref.middle[int(choices[i,j])](z,pos,torch.zeros_like(pos));per.append(q+gates[i,j]*(v-z))
     parts.append(torch.cat(per))
    want=torch.cat(parts,1);torch.testing.assert_close(got,want,rtol=1e-5,atol=1e-6);got.square().mean().backward();want.square().mean().backward();torch.testing.assert_close(h.grad,hr.grad,rtol=2e-5,atol=2e-6)
    for (n,p),(nn,pp) in zip(m.named_parameters(),ref.named_parameters()):
     if p.grad is not None:torch.testing.assert_close(p.grad,pp.grad,rtol=2e-4,atol=2e-6)
    if film:self.assertGreater(float(m.film_gamma.grad.abs().sum()),0);self.assertGreater(float(m.film_beta.grad.abs().sum()),0)
    else:self.assertIsNone(m.film_gamma.grad)
    canonical=copy.deepcopy(m);canonical.reference=True
    a=m.hidden(x);b=canonical.hidden(x);torch.testing.assert_close(a,b,rtol=2e-5,atol=3e-6)
    for cut in (1,3,5,8,12):
     z=x.clone();z[:,cut:]=(z[:,cut:]+7)%37;torch.testing.assert_close(b[:,:cut],canonical.hidden(z)[:,:cut],rtol=0,atol=0)
    a=m.hidden(x)
    for cut in (1,3,5,8,12):
     z=x.clone();z[:,cut:]=(z[:,cut:]+7)%37;torch.testing.assert_close(a[:,:cut],m.hidden(z)[:,:cut],rtol=0,atol=0)
if __name__=='__main__':unittest.main()
