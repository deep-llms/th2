import copy,tempfile,unittest
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from efficient_llm.warm_order.model import Model as Dense,Config
from efficient_llm.warm_order.recursion_model import Model,PATHS,path_for
from efficient_llm.warm_order.recursion_train import train
from efficient_llm.warm_order.data import sha

class RecursionTests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(1);self.c=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=37,length=12,prefix=3)
 def test_init_causality_grouped_gradients_and_budget(self):
  torch.manual_seed(0);dense=Dense(self.c);torch.manual_seed(0);m=Model(self.c)
  for k,v in m.state_dict().items():torch.testing.assert_close(v,dense.state_dict()[k],atol=0,rtol=0)
  x=torch.randint(37,(3,12));alter=x.clone();alter[:,7:]=(alter[:,7:]+1)%37
  for path in PATHS:torch.testing.assert_close(m.hidden(x,path)[:,:7],m.hidden(alter,path)[:,:7],atol=0,rtol=0)
  torch.testing.assert_close(m.features(x),m.features(torch.cat([x[:,:3],alter[:,3:]],1)),atol=0,rtol=0)
  a=copy.deepcopy(m);b=copy.deepcopy(m);choices=torch.tensor([0,2,1]);out=a.routed_nll(x,choices);reference=torch.cat([b.nll(x[i:i+1],PATHS[int(k)]) for i,k in enumerate(choices)])
  torch.testing.assert_close(out,reference,atol=1e-6,rtol=1e-6);out.mean().backward();reference.mean().backward()
  for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=1e-5)
  self.assertEqual(sum(2+len(path_for('RV',i,12)) for i in range(12)),72)
  with self.assertRaises(ValueError):m.hidden(x,(1,0))
 def test_all_arms_fullstate_resume(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);stem=str(r/'data');np.random.default_rng(4).integers(0,37,(16,12),dtype=np.uint32).tofile(stem+'.bin');st=np.zeros((16,12),dtype='u1');st[:,0]=1;st.tofile(stem+'.starts');np.save(r/'ids.npy',np.arange(16))
   for arm in ('D4','RF','RV'):
    c=dict(model=asdict(self.c),arm=arm,steps=8,batch=2,seed=1,lr=.001,warmup=2,data=stem,indices=str(r/'ids.npy'),save_every=2,log_every=1)
    c['expected_inputs']={p:sha(p) for p in (stem+'.bin',stem+'.starts',c['indices'])}
    train(c,r/(arm+'full'),'cpu');train(c,r/(arm+'part'),'cpu',stop_after=3);train(c,r/(arm+'part'),'cpu',resume=True)
    a=torch.load(r/(arm+'full')/'fullstate.pt',weights_only=False);b=torch.load(r/(arm+'part')/'fullstate.pt',weights_only=False)
    for k,v in a['model'].items():torch.testing.assert_close(v,b['model'][k],atol=0,rtol=0)
    self.assertEqual(a['counts'],b['counts']);self.assertEqual(a['block_tokens'],b['block_tokens'])
    for k,s in a['optimizer']['state'].items():
     for f,v in s.items():torch.testing.assert_close(v,b['optimizer']['state'][k][f],atol=0,rtol=0)
    with self.assertRaises(ValueError):train(c,r/(arm+'part'),'cpu',resume=True)
if __name__=='__main__':unittest.main()
