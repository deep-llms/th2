import tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from efficient_llm.warm_order.model import Config,Model,Router,ORDERS,path_for,router_loss
from efficient_llm.warm_order.data import Rows

class Core(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(17)
        self.c=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=37,prefix=3,length=12)
        self.m=Model(self.c).eval();self.x=torch.randint(0,37,(3,12))
    def test_balanced_orders(self):
        self.assertEqual(len(set(ORDERS)),8)
        for j in range(4):self.assertEqual(sorted([p[j] for p in ORDERS]),[0,0,1,1,2,2,3,3])
    def test_curriculum(self):
        self.assertEqual([len(path_for('W',x,100)) for x in [0,9,10,19,20,99]],[1,1,2,2,4,4])
        for a in ['D','R']:
            self.assertTrue(all(len(path_for(a,s,100))==4 for s in range(100)))
    def test_causal_features_and_hidden(self):
        changed=self.x.clone();changed[:,7:]=(changed[:,7:]+1)%37
        self.assertTrue(torch.equal(self.m.features(self.x),self.m.features(changed)))
        for path in ORDERS:
            torch.testing.assert_close(self.m.hidden(self.x,path)[:,:7],self.m.hidden(changed,path)[:,:7],atol=0,rtol=0)
    def test_loss_shift(self):
        h=self.m.hidden(self.x,ORDERS[0]);logits=h@self.m.embed.weight.T
        expected=torch.nn.functional.cross_entropy(logits[:,2:-1].reshape(-1,37),self.x[:,3:].reshape(-1),reduction='none').reshape(3,-1).mean(1)
        torch.testing.assert_close(self.m.nll(self.x,ORDERS[0]),expected)
    def test_grouped_matches_singletons_gradients(self):
        choices=torch.tensor([0,3,7]);self.m.train()
        loss=self.m.routed_nll(self.x,choices).mean();loss.backward()
        grads={k:p.grad.clone() for k,p in self.m.named_parameters()};self.m.zero_grad()
        ref=torch.stack([self.m.nll(self.x[i:i+1],ORDERS[k]).squeeze() for i,k in enumerate(choices)]).mean();ref.backward()
        torch.testing.assert_close(loss,ref)
        for k,p in self.m.named_parameters():torch.testing.assert_close(p.grad,grads[k],atol=2e-6,rtol=1e-4)
    def test_router_updates_only_router(self):
        r=Router(16);cost=torch.rand(3,8,requires_grad=True)
        router_loss(r(self.m.features(self.x)),cost).backward()
        self.assertIsNone(cost.grad);self.assertTrue(all(p.grad is None for p in self.m.parameters()))
        self.assertTrue(any(p.grad.abs().sum()>0 for p in r.parameters()))
    def test_rows_filter_and_refuse_cross_document(self):
        with tempfile.TemporaryDirectory() as d:
            stem=Path(d)/'train';a=np.arange(36,dtype='<u4').reshape(3,12);b=np.zeros((3,12),dtype='u1');b[:,0]=1;b[1,5]=1
            a.tofile(str(stem)+'.bin');b.tofile(str(stem)+'.starts');r=Rows(stem,12)
            np.testing.assert_array_equal(r.eligible(),[0,2])
            self.assertEqual(tuple(r.batch([0,2],'cpu').shape),(2,12))
            with self.assertRaises(ValueError):r.batch([1],'cpu')
    def test_route_rejection(self):
        with self.assertRaises(ValueError):self.m.hidden(self.x,[0,0])
        with self.assertRaises(ValueError):self.m.routed_nll(self.x,torch.tensor([0,1,8]))
