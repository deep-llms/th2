import json,tempfile,unittest
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from efficient_llm.warm_order.model import Config,Model
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.auxiliary_prepare import prepare,NAMES
from efficient_llm.warm_order.auxiliary_eval import run
from efficient_llm.warm_order.auxiliary_report import report,replay_run
from scripts.verify_auxiliary_export import verify

class Tests(unittest.TestCase):
 def test_eight_real_fits_router_report_backup_and_tamper(self):
  torch.set_num_threads(1);c=Config(d=8,heads=1,kv_heads=1,head_dim=8,ff=16,vocab=19,length=8,prefix=2)
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);stems={};hashes={}
   for domain in ('fit','dclm','fwe'):
    s=str(r/domain);stems[domain]=s;np.random.default_rng(9).integers(0,c.vocab,(240,c.length),dtype=np.uint32).tofile(s+'.bin');st=np.zeros((240,c.length),dtype='u1');st[:,0]=1;st[220:,3]=1;st.tofile(s+'.starts');hashes.update({s+e:sha(s+e) for e in ('.bin','.starts')})
   np.savez(r/'eval.npz',fit=np.arange(8),valid=np.arange(8,12),dclm=np.arange(4),fwe=np.arange(4));hashes[str(r/'eval.npz')]=sha(r/'eval.npz')
   old=r/'previous';old.mkdir();np.save(old/'train-s0.npy',np.arange(12,16));np.save(old/'train-s1.npy',np.arange(16,20));(old/'complete.json').write_text(json.dumps(dict(files={p.name:sha(p) for p in old.iterdir()})))
   torch.save(dict(config=asdict(c),model=Model(c).state_dict()),r/'initial.pt');hashes[str(r/'initial.pt')]=sha(r/'initial.pt');hashes[str(old/'complete.json')]=sha(old/'complete.json')
   plan=dict(hashes=hashes,training={},previous_inputs=str(old/'complete.json'),weights={str(s):str(r/'initial.pt') for s in (0,1)},eval=dict(stems=stems,indices=str(r/'eval.npz')))
   for seed in (0,1):plan['training'][str(seed)]=dict(model=asdict(c),steps=80,batch=2,seed=seed,lr=.0003,warmup=4,router_warmup=20,save_every=40,log_every=40,data=stems['fit'],threads=1)
   base=r/'study';prepare(plan,base/'inputs');(base/'execution').mkdir()
   for p in ('source.json','admission.json'):(base/'execution'/p).write_text('{}')
   for seed in (0,1):self.assertTrue(np.load(base/f'inputs/train-s{seed}.npy').max()<220)
   for name in NAMES:run(json.loads((base/f'inputs/{name}.json').read_text()),base/'runs'/name,'cpu')
   result=report(base,r/'backup');self.assertEqual(result['node_raw_replay'],'PASS');self.assertEqual(len(json.loads((base/'backup.json').read_text())['node_copies']),17);self.assertEqual(verify(base/'results.tar.gz',base/'backup.json',r/'verified')['status'],'PASS')
   p=base/'runs/A-s0/dclm/summary.npz';a=dict(np.load(p));a['choices'][0]=(a['choices'][0]+1)%2;np.savez(p,**a)
   with self.assertRaises(ValueError):replay_run(base/'runs/A-s0')
