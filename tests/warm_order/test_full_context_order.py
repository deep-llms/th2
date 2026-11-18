import unittest,copy
import torch
from efficient_llm.warm_order.model import Config,Model as Dense
from efficient_llm.warm_order.full_context_order import Model,unit_residual
from efficient_llm.warm_order.window_order import bounds

class FullContextTests(unittest.TestCase):
 def test_dense_identity_causality_and_cross_window_context(self):
  torch.set_num_threads(1);torch.manual_seed(61)
  c=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=37,length=13,prefix=3)
  dense=Dense(c).eval();x=torch.randint(37,(2,13));ref=dense.hidden(x,(0,1,2,3))
  for mode in ('F','H','S'):
   m=Model(c,mode,5,2).eval();m.load_dense(dense.state_dict());y=m.hidden(x)
   torch.testing.assert_close(y,ref,rtol=0,atol=0)
   # Force diverse routes, then independently test strict future independence.
   with torch.no_grad():m.router.weight.normal_(std=.5);m.depth_prior.zero_()
   y=m.hidden(x);old=[v.clone() for v in m.last_routes]
   for cut in (1,3,5,8,12):
    z=x.clone();z[:,cut:]=(z[:,cut:]+7)%37
    torch.testing.assert_close(m.hidden(z)[:,:cut],y[:,:cut],rtol=0,atol=0)
    for depth in range(4):
     for j,(a,b) in enumerate(bounds(13,5,(depth*2)%5)):
      if a<cut:torch.testing.assert_close(m.last_routes[depth][:,j],old[depth][:,j],rtol=0,atol=0)
   # Inspect one middle block, independent of stem/coda and route selection.
   h=torch.randn(2,13,16);pos=torch.arange(13).expand(2,-1);seg=torch.zeros_like(pos)
   a=m.middle[0](h,pos,seg);changed=h.clone();changed[:,0]+=torch.arange(16)/4
   b=m.middle[0](changed,pos,seg);self.assertGreater(float((a[:,9]-b[:,9]).abs().max()),1e-5)
   for i in range(2):torch.testing.assert_close(m.hidden(x[i:i+1]),m.hidden(x)[i:i+1],rtol=1e-5,atol=1e-6)

 def test_independent_unit_forward_and_analytic_gradient(self):
  torch.manual_seed(5);z=torch.randn(2,3,4,dtype=torch.float64,requires_grad=True);p=z.softmax(-1);ch=p.argmax(-1);d=torch.randn(2,3,4,5,dtype=torch.float64,requires_grad=True);u=torch.randn(2,3,5,dtype=torch.float64)
  hard=unit_residual(p,ch,d,'F');mask=torch.nn.functional.one_hot(ch,4)
  for mode in ('H','S'):
   out=unit_residual(p,ch,d,mode);torch.testing.assert_close(out,hard,rtol=0,atol=0)
   gz,gd=torch.autograd.grad((out*u).sum(),(z,d),retain_graph=True)
   gp=(d.detach()*u[:,:,None]).sum(-1)*(mask if mode=='H' else 1)
   expected=p.detach()*(gp-(p.detach()*gp).sum(-1,keepdim=True))
   torch.testing.assert_close(gz,expected,rtol=1e-12,atol=1e-12)
   torch.testing.assert_close(gd,mask[...,None]*u[:,:,None],rtol=0,atol=0)

if __name__=='__main__':unittest.main()
