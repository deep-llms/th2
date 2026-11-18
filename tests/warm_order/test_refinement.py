import unittest
import torch
from efficient_llm.warm_order.model import Config,Model,ORDERS
from efficient_llm.warm_order.refinement import Refinement,ExitRouter,sparse_block,canonical_state

class RefinementTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(19)
        self.c=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=37,prefix=3,length=12)
        self.base=Model(self.c);self.x=torch.randint(0,37,(3,12))
    def test_identity_initialization_and_gradient(self):
        m=Refinement(self.base,frozen=True);a,b=m.states(self.x)
        torch.testing.assert_close(a,self.base.hidden(self.x,ORDERS[0]),rtol=0,atol=0)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        b.square().sum().backward()
        self.assertTrue(all(p.grad is None for p in self.base.parameters()))
        self.assertGreater(float(m.extra.self_attn.o_proj.weight.grad.abs().sum()),0)
        self.assertGreater(float(m.extra.mlp.down_proj.weight.grad.abs().sum()),0)
    def test_sparse_dense_outputs_and_gradients(self):
        h=torch.randn(3,12,16,requires_grad=True);block=self.base.coda
        pos=torch.arange(12).expand(3,-1);seg=torch.zeros(3,12,dtype=torch.long)
        for mask in (torch.ones(3,12,dtype=torch.bool),torch.zeros(3,12,dtype=torch.bool),torch.rand(3,12)>.5):
            sparse=sparse_block(block,h,mask);full=block(h,pos,seg);ref=torch.where(mask[...,None],full,h)
            torch.testing.assert_close(sparse,ref,rtol=1e-5,atol=1e-6)
            params=[h,*block.parameters()]
            ga=torch.autograd.grad(sparse.square().sum(),params,allow_unused=True,retain_graph=True)
            gb=torch.autograd.grad(ref.square().sum(),params,allow_unused=True,retain_graph=True)
            for p,a,b in zip(params,ga,gb):torch.testing.assert_close(torch.zeros_like(p) if a is None else a,torch.zeros_like(p) if b is None else b,rtol=3e-4,atol=1e-4)
    def test_future_independence_and_no_target_router(self):
        h=canonical_state(self.base,self.x);router=ExitRouter(16)
        changed=self.x.clone();changed[:,6:]=(changed[:,6:]+1)%37;hc=canonical_state(self.base,changed)
        mask=router(h).argmax(-1).bool();maskc=router(hc).argmax(-1).bool()
        torch.testing.assert_close(h[:,:6],hc[:,:6],rtol=0,atol=0);self.assertTrue(torch.equal(mask[:,:6],maskc[:,:6]))
        y=sparse_block(self.base.coda,h,mask);yc=sparse_block(self.base.coda,hc,maskc)
        torch.testing.assert_close(y[:,:6],yc[:,:6],rtol=1e-5,atol=1e-6)
        router(h).sum().backward();self.assertTrue(all(p.grad is None for p in self.base.parameters()))
    def test_zero_mask_no_extra_calls(self):
        h=torch.randn(2,12,16);calls=[]
        handles=[m.register_forward_hook(lambda *args:calls.append(1)) for m in (self.base.coda.self_attn.q_proj,self.base.coda.self_attn.k_proj,self.base.coda.mlp)]
        out=sparse_block(self.base.coda,h,torch.zeros(2,12,dtype=torch.bool));self.assertEqual(calls,[]);torch.testing.assert_close(h,out,rtol=0,atol=0)
        for x in handles:x.remove()
if __name__=='__main__':unittest.main()
