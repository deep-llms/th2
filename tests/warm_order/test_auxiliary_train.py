import json,tempfile,unittest
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch

from efficient_llm.warm_order.model import Config,Model,ORDERS
from efficient_llm.warm_order.auxiliary_train import train
from efficient_llm.warm_order.data import sha

class Tests(unittest.TestCase):
 def test_resume_crosses_stage_boundary_exactly(self):
  torch.set_num_threads(1);c=Config(d=8,heads=1,kv_heads=1,head_dim=8,ff=16,vocab=19,length=8,prefix=2)
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);stem=str(r/'data');np.random.default_rng(2).integers(0,c.vocab,(160,c.length),dtype=np.uint32).tofile(stem+'.bin');st=np.zeros((160,c.length),dtype='u1');st[:,0]=1;st.tofile(stem+'.starts');np.save(r/'ids.npy',np.arange(160))
   cfg=dict(model=asdict(c),arm='A',seed=0,batch=2,steps=80,lr=.0003,warmup=4,router_warmup=20,data=stem,indices=str(r/'ids.npy'),log_every=20,save_every=20,threads=1);torch.save(dict(config=asdict(c),model=Model(c).state_dict()),r/'initial.pt');cfg['initial_weights']=str(r/'initial.pt');cfg['expected_inputs']={p:sha(p) for p in (stem+'.bin',stem+'.starts',cfg['indices'],cfg['initial_weights'])}
   a,ao=train(cfg,r/'whole','cpu');train(cfg,r/'resume','cpu',stop_after=19);b,bo=train(cfg,r/'resume','cpu',resume=True)
   for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
   for key,value in ao.state_dict()['state'].items():
    for n,v in value.items():torch.testing.assert_close(v,bo.state_dict()['state'][key][n],rtol=0,atol=0)
   one=torch.load(r/'whole/fullstate.pt',weights_only=False);two=torch.load(r/'resume/fullstate.pt',weights_only=False)
   torch.testing.assert_close(one['torch_rng'],two['torch_rng'],rtol=0,atol=0);self.assertEqual(one['counts'],two['counts']);self.assertEqual(one['block_tokens'],two['block_tokens']);self.assertEqual(one['block_tokens'],80*(12*2*8+2*c.prefix))
   bad=dict(cfg,lr=.002)
   with self.assertRaises(ValueError):train(bad,r/'resume','cpu',resume=True)
