import json,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from efficient_llm.warm_order.model import Config,Model
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.window_order_prepare import prepare as old_prepare,NAMES
from efficient_llm.warm_order.window_order_eval import run as old_run
from efficient_llm.warm_order.window_recovery_prepare import prepare
from efficient_llm.warm_order.window_recovery_eval import run
from efficient_llm.warm_order.window_recovery_report import report,replay_run
from scripts.verify_window_recovery_export import verify

class RecoveryTests(unittest.TestCase):
 def test_eval_only_ten_jobs_unchanged_parents_exact_reference_export_backups(self):
  torch.set_num_threads(1);torch.manual_seed(4);c=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=37,length=12,prefix=3)
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);stems={};hashes={}
   for domain in ('fit','dclm','fwe'):
    s=str(r/domain);stems[domain]=s;np.random.default_rng(9).integers(0,37,(40,12),dtype=np.uint32).tofile(s+'.bin');st=np.zeros((40,12),dtype='u1');st[:,0]=1;st.tofile(s+'.starts');hashes.update({s+e:sha(s+e) for e in ('.bin','.starts')})
   np.save(r/'train.npy',np.arange(8));np.savez(r/'eval.npz',fit=np.arange(8,16),valid=np.arange(16,20),dclm=np.arange(4),fwe=np.arange(4));hashes[str(r/'eval.npz')]=sha(r/'eval.npz');torch.save(dict(config=asdict(c),model=Model(c).state_dict()),r/'initial.pt');hashes[str(r/'initial.pt')]=sha(r/'initial.pt')
   plan=dict(hashes=hashes,training={},weights={str(s):str(r/'initial.pt') for s in (0,1)},window=5,stride=2,eval=dict(stems=stems,indices=str(r/'eval.npz')))
   for seed in (0,1):
    tc=dict(model=asdict(c),steps=4,batch=2,seed=seed,lr=.0003,warmup=1,save_every=2,log_every=2,data=stems['fit'],indices=str(r/'train.npy'));tc['expected_inputs']={p:sha(p) for p in (stems['fit']+'.bin',stems['fit']+'.starts',tc['indices'])};plan['hashes'].update(tc['expected_inputs']);plan['training'][str(seed)]=tc
   parent=r/'parent';old_prepare(plan,parent/'inputs');(parent/'execution').mkdir();(parent/'execution/source.json').write_text('{}');(parent/'terminal.json').write_text('{"success":false,"native_verified":true}');(parent/'failure.json').write_text('{"error":"retained failure fixture"}')
   for name in NAMES:old_run(json.loads((parent/f'inputs/{name}.json').read_text()),parent/'runs'/name,'cpu')
   before={str(p):sha(p) for p in parent.rglob('*') if p.is_file()};lineage={k:str(parent/p) for k,p in [('failure','failure.json'),('source','execution/source.json'),('terminal','terminal.json')]};h=dict(plan['hashes']);h.update({p:sha(p) for p in lineage.values()})
   for name in NAMES:
    if not name.startswith('D6'):
     for f in ('weights.pt','fullstate.pt'):
      p=parent/'runs'/name/'train'/f;h[str(p)]=sha(p)
   base=r/'recovery';prepare(dict(parent=str(parent),hashes=h,lineage=lineage),base/'inputs');(base/'execution').mkdir()
   for p in ('source.json','admission.json'):(base/'execution'/p).write_text('{}')
   with patch('efficient_llm.warm_order.window_order_train.train',side_effect=AssertionError('No retraining')):
    for name in NAMES:run(json.loads((base/f'inputs/{name}.json').read_text()),base/'runs'/name,'cpu')
   result=report(base,r/'backup');self.assertEqual(result['node_raw_replay'],'PASS');self.assertEqual(len(json.loads((base/'backup.json').read_text())['node_copies']),17);self.assertEqual(verify(base/'results.tar.gz',base/'backup.json',r/'verified')['status'],'PASS')
   self.assertEqual(before,{str(p):sha(p) for p in parent.rglob('*') if p.is_file()})
   p=base/'runs/WL-s0/dclm/summary.npz';a=dict(np.load(p));a['nll'][0]+=.01;np.savez(p,**a)
   with self.assertRaises(ValueError):replay_run(base/'runs/WL-s0')
if __name__=='__main__':unittest.main()
