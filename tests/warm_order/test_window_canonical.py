import unittest
import torch
from efficient_llm.warm_order.model import Config
from efficient_llm.warm_order.window_order import Model as Grouped
from efficient_llm.warm_order.window_canonical import Model

class ReferenceTests(unittest.TestCase):
 def test_fixed_shape_equivalence_gradients_routes_and_exact_causality(self):
  torch.set_num_threads(1);torch.manual_seed(21);c=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=37,length=13,prefix=3);x=torch.randint(37,(3,13))
  for arm in ('WF','WR','WG','WL'):
   a=Grouped(c,arm,5,2).eval();b=Model(c,arm,5,2).eval()
   with torch.no_grad():a.router.weight.normal_(std=.2);a.router.bias.normal_(std=.1)
   b.load_state_dict(a.state_dict());ha=a.hidden(x);hb=b.hidden(x);torch.testing.assert_close(ha,hb,rtol=2e-5,atol=3e-6)
   for p,q in zip(a.last_routes,b.last_routes):torch.testing.assert_close(p,q,rtol=0,atol=0)
   ha.square().mean().backward();hb.square().mean().backward()
   for p,q in zip(a.parameters(),b.parameters()):
    if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=3e-4,atol=3e-6)
   for cut in (1,3,5,8,12):
    z=x.clone();z[:,cut:]=(z[:,cut:]+7)%37;torch.testing.assert_close(hb[:,:cut],b.hidden(z)[:,:cut],rtol=0,atol=0)
if __name__=='__main__':unittest.main()
