import unittest,copy
import torch
from efficient_llm.warm_order.model import Config
from efficient_llm.warm_order.auxiliary_order import Model,PATHS
class Tests(unittest.TestCase):
 def test_gradient_separation_and_exploration(self):
  torch.set_num_threads(1);torch.manual_seed(7);c=Config(d=8,heads=1,kv_heads=1,head_dim=8,ff=16,vocab=19,length=8,prefix=2);x=torch.randint(19,(3,8))
  for arm in ('D','U','G','A'):
   m=Model(c,arm);body,aux=m.objective(x,3,2)
   aux.backward(retain_graph=True);self.assertTrue(all(p.grad is None for p in m.body.parameters()));m.zero_grad();body.backward();self.assertTrue(all(p.grad is None for p in m.head.parameters()))
   scores=torch.tensor([[.2,-.1],[-.1,.2],[0.,0.]]);w=m.weights(scores,3,2);torch.testing.assert_close(w.sum(1),torch.ones(3))
   if arm in ('G','A'):self.assertGreaterEqual(float(w.min()),.04999)
   m.eval();changed=x.clone();changed[:,2:]=(changed[:,2:]+1)%19
   self.assertTrue(torch.equal(m.scores(x),m.scores(changed)))
   with torch.no_grad():torch.testing.assert_close(m.routed_nll(x,torch.tensor([0,1,0])),m.costs(x)[torch.arange(3),torch.tensor([0,1,0])],atol=1e-6,rtol=0)
 def test_global_head_and_body_control(self):
  torch.set_num_threads(1);c=Config(d=8,heads=1,kv_heads=1,head_dim=8,ff=16,vocab=19,length=8,prefix=2);m=Model(c,'G');f=torch.randn(5,8);self.assertTrue(torch.equal(m.head(f)[0],m.head(f)[4]));d=Model(c,'D');x=torch.randint(19,(2,8));b,_=d.objective(x,10,2);torch.testing.assert_close(b,d.body.nll(x,PATHS[0]).mean(),atol=0,rtol=0)
