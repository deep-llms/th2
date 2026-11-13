import copy, json, tempfile, unittest
from pathlib import Path
import numpy as np
import torch
from efficient_llm.dot_pool.rl import *
from efficient_llm.dot_pool.rl_run import update, optim, evaluate
from efficient_llm.dot_pool.train import seed

def tiny():return Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=2240,stem=0,coda=0)
def rows(n=8):return np.array([[i,2000+i,2008+i,(i+2)%20] for i in range(n)],dtype=np.int64)

class Core(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_counts(self):
        with torch.device('meta'):m=Library();p=Policy()
        self.assertLess(abs(m.counts()/12.59e6-1),.001);self.assertEqual(sum(x.numel() for x in p.parameters()),16436)
    def test_layers_equal_existing_loop(self):
        from efficient_llm.dot_pool.model import Model
        seed(1);c=tiny();c.arm='L';old=Model(c);m=Library(c)
        m.embed_tokens.load_state_dict(old.embed_tokens.state_dict());m.norm.load_state_dict(old.norm.state_dict());m.layers.load_state_dict(old.middle.state_dict())
        x,pos,y=batch(rows(),torch.device('cpu'));a=m(x,pos,mode='loop')['logits'];b=old(x)[torch.arange(len(x)),pos]
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    def test_leakage(self):
        seed(1);m=Library(tiny());p=Policy(16)
        with torch.no_grad():p.W.normal_()
        x,pos,_=batch(rows(),torch.device('cpu'));z=x.clone();z[torch.arange(len(x)),pos+1]=(z[torch.arange(len(x)),pos+1]+10)%2000
        seed(12);a=m(x,pos,p);seed(12);b=m(z,pos,p)
        torch.testing.assert_close(a['logits'],b['logits'],rtol=0,atol=0);self.assertTrue(torch.equal(a['paths'],b['paths']))
    def test_gradient_isolation(self):
        seed(2);m=Library(tiny());p=Policy(16);x,pos,y=batch(rows(),torch.device('cpu'))
        ce=F.cross_entropy(m(x,pos,p)['logits'],y);ce.backward()
        self.assertTrue(all(v.grad is None for v in p.parameters()))
        before=[v.grad.clone() for v in m.parameters()]
        loss,_=rollout(m,p,rows(),torch.device('cpu'));loss.backward()
        self.assertTrue(all(torch.equal(a,v.grad) for a,v in zip(before,m.parameters())))
        self.assertTrue(any(v.grad is not None and v.grad.abs().sum()>0 for v in p.parameters()))
        m.zero_grad();p.zero_grad();loss,_=rollout(m,p,rows(),torch.device('cpu'));loss.backward()
        self.assertTrue(all(v.grad is None for v in m.parameters()))
    def test_grouping(self):
        m=Library(tiny());x,pos,_=batch(rows(),torch.device('cpu'));seed(4);paths=torch.randint(4,(8,8));a=m(x,pos,paths=paths)['logits']
        b=torch.cat([m(x[i:i+1],pos[i:i+1],paths=paths[i:i+1])['logits'] for i in range(8)])
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
    def test_advantages(self):
        r=torch.tensor([[1.,2.,3.,4.],[5.,5.,5.,5.]])
        a,s=advantages(r);d=r-r.mean(1,keepdim=True)
        torch.testing.assert_close(a,d/(d.std(unbiased=False)+1e-6));self.assertTrue(torch.equal(a[1],torch.zeros(4)))
        paths=torch.zeros(2,4,8,dtype=torch.long);a,s=advantages(r,paths);self.assertEqual(a.abs().sum(),0)
    def test_resume(self):
        seed(8);m=Library(tiny());p=Policy(16);device=torch.device('cpu');opt,po=optim(m,p,device);stream=RewardStream(8,4)
        cfg=dict(arm='J-RL',seed=8,model={},batch=8,lr=.001,spec_sha256=SPEC_SHA)
        def step(i):
            idx=stream.next(4);return float(update(m,p,opt,po,rows(),rows()[idx],device,i,active=True)['ce'])
        for i in range(1,4):step(i)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'ckpt';save(path,m,p,opt,po,3,cfg,stream);a=[step(i) for i in range(4,14)];h=tensor_hash(m)
            n,_=load(path,m,p,opt,po,cfg,stream,device);b=[step(i) for i in range(4,14)]
            self.assertEqual(n,3);self.assertEqual(a,b);self.assertEqual(h,tensor_hash(m))
    def test_data_disjoint(self):
        with tempfile.TemporaryDirectory() as d:
            root=make_data(Path(d)/'data');v=np.load(root/'split_indices.npz');split=v['split'];self.assertEqual(np.bincount(split).tolist(),[34000,4000,2000])
            keys=['id_dev','id_test','train_compositions','rl_train','rl_dev','ood_dev','ood_test']
            merged=np.concatenate([v[k] for k in keys]);self.assertEqual(len(merged),len(np.unique(merged)))
            for key in keys:
                expect=0 if key in keys[:3] else 1 if key.startswith('rl_') else 2
                self.assertTrue(np.all(split[v['first'][v[key]]]==expect));self.assertTrue(np.all(split[v['second'][v[key]]]==expect))
            self.assertEqual(len(np.load(root/'train.npy')),284800)
    def test_reward_stream(self):
        s=RewardStream(19,42);v=np.concatenate([s.next(7) for _ in range(19)]).reshape(7,19)
        self.assertTrue(all(len(np.unique(x))==19 for x in v))
    def test_budget(self):
        self.assertEqual(choose_budget(8)['epochs'],4500);self.assertEqual(choose_budget(12)['epochs'],3000)
        self.assertEqual(choose_budget(12)['switch_epoch'],1000);self.assertTrue(choose_budget(30)['minimum_overrun'])

if __name__=='__main__':unittest.main()
