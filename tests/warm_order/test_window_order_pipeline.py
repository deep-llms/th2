import json,tempfile,unittest
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from efficient_llm.warm_order.model import Config,Model
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.window_order_prepare import prepare,NAMES
from efficient_llm.warm_order.window_order_eval import run
from efficient_llm.warm_order.window_order_report import report,replay_run
from scripts.verify_window_order_export import verify

class PipelineTests(unittest.TestCase):
 def test_eight_fits_plus_two_dense_evals_eval_report_backup_export_and_tamper(self):
  torch.set_num_threads(1);torch.manual_seed(4);c=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=37,length=12,prefix=3)
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);stems={};hashes={}
   for domain in ('fit','dclm','fwe'):
    s=str(r/domain);stems[domain]=s;np.random.default_rng(9).integers(0,37,(40,12),dtype=np.uint32).tofile(s+'.bin');st=np.zeros((40,12),dtype='u1');st[:,0]=1;st.tofile(s+'.starts');hashes.update({s+e:sha(s+e) for e in ('.bin','.starts')})
   np.save(r/'train.npy',np.arange(8));np.savez(r/'eval.npz',fit=np.arange(8,16),valid=np.arange(16,20),dclm=np.arange(4),fwe=np.arange(4));hashes[str(r/'eval.npz')]=sha(r/'eval.npz');torch.save(dict(config=asdict(c),model=Model(c).state_dict()),r/'initial.pt')
   plan=dict(hashes=hashes,training={},weights={str(s):str(r/'initial.pt') for s in (0,1)},window=5,stride=2,eval=dict(stems=stems,indices=str(r/'eval.npz')))
   for seed in (0,1):
    tc=dict(model=asdict(c),steps=4,batch=2,seed=seed,lr=.0003,warmup=1,save_every=2,log_every=2,data=stems['fit'],indices=str(r/'train.npy'));tc['expected_inputs']={p:sha(p) for p in (stems['fit']+'.bin',stems['fit']+'.starts',tc['indices'])};plan['hashes'].update(tc['expected_inputs']);plan['training'][str(seed)]=tc
   plan['hashes'][str(r/'initial.pt')]=sha(r/'initial.pt')
   base=r/'study';prepare(plan,base/'inputs');(base/'execution').mkdir()
   for p in ('source.json','admission.json'):(base/'execution'/p).write_text('{}')
   for name in NAMES:run(json.loads((base/f'inputs/{name}.json').read_text()),base/'runs'/name,'cpu')
   result=report(base,r/'backup');self.assertEqual(result['node_raw_replay'],'PASS');self.assertEqual(len(json.loads((base/'backup.json').read_text())['node_copies']),17);self.assertEqual(verify(base/'results.tar.gz',base/'backup.json',r/'verified')['status'],'PASS')
   p=base/'runs/WL-s0/dclm/summary.npz';a=dict(np.load(p));a['routes'][0,0]=(int(a['routes'][0,0])+1)%4;np.savez(p,**a)
   with self.assertRaises(ValueError):replay_run(base/'runs/WL-s0')
if __name__=='__main__':unittest.main()
