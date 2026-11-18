import json,tempfile,unittest
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from efficient_llm.warm_order.model import Config,Model as Dense
from efficient_llm.warm_order.window_order import Model as Parent
from efficient_llm.warm_order.window_film_train import train
from efficient_llm.warm_order.data import sha

def equal(a,b):
 if torch.is_tensor(a):torch.testing.assert_close(a,b,rtol=0,atol=0)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b

class ResumeTests(unittest.TestCase):
 def test_all_arms_exact_optimizer_rng_balance_data_resume_and_guard(self):
  c=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=37,length=13,prefix=3)
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);stem=str(r/'data');np.random.default_rng(7).integers(0,37,(30,13),dtype=np.uint32).tofile(stem+'.bin');np.zeros((30,13),dtype='u1').tofile(stem+'.starts');np.save(r/'ids.npy',np.arange(16))
   for arm in ('G0','GF','L0','LF','D6C'):
    parent=Dense(c) if arm=='D6C' else Parent(c,'WG' if arm.startswith('G') else 'WL',5,2,1)
    torch.save(dict(config=asdict(c),model=parent.state_dict(),arm=getattr(parent,'arm','D6'),window=5,stride=2,seed=1),r/'parent.pt')
    tc=dict(initial_weights=str(r/'parent.pt'),model=asdict(c),arm=arm,window=5,stride=2,steps=8,batch=2,seed=1,lr=.003,warmup=2,save_every=4,log_every=1,data=stem,indices=str(r/'ids.npy'),threads=1);tc['expected_inputs']={p:sha(p) for p in (stem+'.bin',stem+'.starts',tc['indices'],tc['initial_weights'])}
    a=r/(arm+'full');b=r/(arm+'resume');train(tc,a,'cpu');train(tc,b,'cpu',stop_after=3);train(tc,b,'cpu',resume=True)
    aa=torch.load(a/'fullstate.pt',weights_only=False);bb=torch.load(b/'fullstate.pt',weights_only=False)
    for k in ('model','optimizer','torch_rng','counts','block_tokens','step'):equal(aa[k],bb[k])
    for p in (a,b):
     t=[json.loads(x) for x in (p/'trace.jsonl').read_text().splitlines()];self.assertEqual([x['first_row'] for x in t],list(range(0,16,2)))
     if 'loads' in aa['model']:self.assertEqual(float(aa['model']['loads'].sum()),0)
    if arm.endswith('0'):self.assertEqual(float(aa['model']['film_gamma'].abs().sum()),0)
    elif arm.endswith('F'):self.assertGreater(float(aa['model']['film_gamma'].abs().sum()),0)
    self.assertEqual(json.loads((b/'resume.json').read_text())['resumed_step'],3)
    with self.assertRaises(ValueError):train(tc,b,'cpu',resume=True)
    with self.assertRaises(ValueError):train(tc,b,'cpu')
   bad=dict(tc,lr=.004);cut=r/'bad';train(tc,cut,'cpu',stop_after=3)
   with self.assertRaises(ValueError):train(bad,cut,'cpu',resume=True)
if __name__=='__main__':unittest.main()
