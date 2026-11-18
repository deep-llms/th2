import unittest,copy
import torch
from efficient_llm.warm_order.model import Config,Model
from efficient_llm.warm_order.refinement_vectorized import sparse_block,physical_work

class VectorizedTests(unittest.TestCase):
 def test_outputs_gradients_ragged_masks_and_causality(self):
  torch.set_num_threads(1);torch.manual_seed(817)
  c=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=37,prefix=3,length=12)
  for kind in ('none','all','ragged'):
   b=Model(c).coda;s=copy.deepcopy(b);h=torch.randn(4,12,16,requires_grad=True);hs=h.detach().clone().requires_grad_()
   mask=torch.zeros(4,12,dtype=torch.bool) if kind=='none' else torch.ones(4,12,dtype=torch.bool)
   if kind=='ragged':mask[0]=False;mask[1,1:]=False;mask[2,::2]=False;mask[3,:4]=False
   pos=torch.arange(12).expand(4,-1);dense=b(h,pos,torch.zeros_like(pos));expected=torch.where(mask[:,:,None],dense,h);got=sparse_block(s,hs,mask)
   torch.testing.assert_close(got,expected,atol=2e-6,rtol=2e-6)
   expected.square().sum().backward();got.square().sum().backward();torch.testing.assert_close(h.grad,hs.grad,atol=3e-6,rtol=3e-6)
   for p,q in zip(b.parameters(),s.parameters()):
    if q.grad is None:self.assertTrue(p.grad is None or not torch.count_nonzero(p.grad))
    else:torch.testing.assert_close(p.grad,q.grad,atol=2e-5,rtol=2e-5)
   changed=hs.detach().clone();changed[:,7:]+=10
   torch.testing.assert_close(sparse_block(s,changed,mask)[:,:7],got.detach()[:,:7],atol=0,rtol=0)
   work=physical_work(mask);self.assertGreaterEqual(work['padded_queries'],work['logical_queries'])
if __name__=='__main__':unittest.main()
