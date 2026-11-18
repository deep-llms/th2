import tempfile,unittest
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from efficient_llm.warm_order.model import Model,Config
from efficient_llm.warm_order.order_distill_train import train
from efficient_llm.warm_order.order_distill import ARMS
from efficient_llm.warm_order.data import sha

class TrainerTests(unittest.TestCase):
 def test_all_four_arms_exact_resume_and_locks(self):
  torch.set_num_threads(1);torch.manual_seed(5)
  cfg=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=37,length=12,prefix=3)
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);stem=str(r/'data');np.random.default_rng(2).integers(0,37,(12,12),dtype=np.uint32).tofile(stem+'.bin');start=np.zeros((12,12),dtype='u1');start[:,0]=1;start.tofile(stem+'.starts');np.save(r/'ids.npy',np.arange(8));torch.save(dict(config=asdict(cfg),model=Model(cfg).state_dict()),r/'initial.pt');initial=sha(r/'initial.pt')
   c=dict(model=asdict(cfg),steps=4,batch=2,seed=0,lr=.0003,warmup=1,save_every=2,log_every=2,data=stem,indices=str(r/'ids.npy'),initial_weights=str(r/'initial.pt'))
   c['expected_inputs']={p:sha(p) for p in (stem+'.bin',stem+'.starts',c['indices'],c['initial_weights'])}
   for arm in ARMS:
    a=dict(c,arm=arm);full,opt=train(a,r/(arm+'-full'),'cpu');train(a,r/(arm+'-resume'),'cpu',stop_after=2);res,optr=train(a,r/(arm+'-resume'),'cpu',resume=True)
    for k,v in full.state_dict().items():self.assertTrue(torch.equal(v,res.state_dict()[k]),k)
    for k,v in opt.state_dict()['state'].items():
     for n,z in v.items():
      zz=optr.state_dict()['state'][k][n]
      self.assertTrue(torch.equal(z,zz) if torch.is_tensor(z) else z==zz,n)
    self.assertEqual(sha(r/'initial.pt'),initial)
    with self.assertRaises(ValueError):train(a,r/(arm+'-resume'),'cpu',resume=True)
   bad=dict(c,arm='PK2',expected_inputs={})
   with self.assertRaises(ValueError):train(bad,r/'bad','cpu')
if __name__=='__main__':unittest.main()
