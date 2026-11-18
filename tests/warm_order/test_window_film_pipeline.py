import json,tempfile,unittest
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from efficient_llm.warm_order.model import Config,Model
from efficient_llm.warm_order.window_order import Model as Parent
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.window_film_prepare import prepare,NAMES
from efficient_llm.warm_order.window_film_eval import run
from efficient_llm.warm_order.window_film_report import report,replay_run
from scripts.verify_window_film_export import verify

class PipelineTests(unittest.TestCase):
 def test_ten_continuations_eval_report_backup_export_and_tamper(self):
  torch.set_num_threads(1);torch.manual_seed(4);c=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=37,length=12,prefix=3)
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);stems={};hashes={}
   for domain in ('fit','dclm','fwe'):
    s=str(r/domain);stems[domain]=s;np.random.default_rng(9).integers(0,37,(40,12),dtype=np.uint32).tofile(s+'.bin');st=np.zeros((40,12),dtype='u1');st[:,0]=1;st.tofile(s+'.starts');hashes.update({s+e:sha(s+e) for e in ('.bin','.starts')})
   np.save(r/'previous.npy',np.arange(20,24));hashes[str(r/'previous.npy')]=sha(r/'previous.npy');np.save(r/'train.npy',np.arange(8));np.savez(r/'eval.npz',fit=np.arange(8,16),valid=np.arange(16,20),dclm=np.arange(4),fwe=np.arange(4));hashes[str(r/'eval.npz')]=sha(r/'eval.npz');torch.save(dict(config=asdict(c),model=Model(c).state_dict()),r/'initial.pt')
   (r/'parent-report.json').write_text(json.dumps(dict(continuation_all_seeds_domains=False)));hashes[str(r/'parent-report.json')]=sha(r/'parent-report.json')
   plan=dict(parent_report=str(r/'parent-report.json'),hashes=hashes,training={},weights={k:{} for k in ('WG','WL','D6')},previous_indices=[str(r/'previous.npy')],window=5,stride=2,eval=dict(stems=stems,indices=str(r/'eval.npz')))
   for seed in (0,1):
    for arm in ('WG','WL','D6'):
     m=Model(c) if arm=='D6' else Parent(c,arm,5,2,seed)
     p=r/f'{arm}-s{seed}.pt';torch.save(dict(config=asdict(c),model=m.state_dict(),arm=arm,window=5,stride=2,seed=seed),p);plan['weights'][arm][str(seed)]=str(p);plan['hashes'][str(p)]=sha(p)
    tc=dict(model=asdict(c),steps=4,batch=2,seed=seed,lr=.0003,warmup=1,save_every=2,log_every=2,data=stems['fit'],indices=str(r/'train.npy'));tc['expected_inputs']={p:sha(p) for p in (stems['fit']+'.bin',stems['fit']+'.starts',tc['indices'])};plan['hashes'].update(tc['expected_inputs']);plan['training'][str(seed)]=tc
   plan['hashes'][str(r/'initial.pt')]=sha(r/'initial.pt')
   # A positive parent screen must not activate this negative-screen follow-up.
   pp=r/'parent-report.json';pp.write_text(json.dumps(dict(continuation_all_seeds_domains=True)));plan['hashes'][str(pp)]=sha(pp)
   with self.assertRaises(ValueError):prepare(plan,r/'forbidden-inputs')
   self.assertFalse((r/'forbidden-inputs').exists())
   pp.write_text(json.dumps(dict(continuation_all_seeds_domains=False)));plan['hashes'][str(pp)]=sha(pp)
   base=r/'study';prepare(plan,base/'inputs');(base/'execution').mkdir()
   for p in ('source.json','admission.json'):(base/'execution'/p).write_text('{}')
   for name in NAMES:run(json.loads((base/f'inputs/{name}.json').read_text()),base/'runs'/name,'cpu')
   result=report(base,r/'backup');self.assertEqual(result['node_raw_replay'],'PASS');self.assertEqual(len(json.loads((base/'backup.json').read_text())['node_copies']),21);self.assertEqual(verify(base/'results.tar.gz',base/'backup.json',r/'verified')['status'],'PASS')
   p=base/'runs/LF-s0/dclm/summary.npz';a=dict(np.load(p));a['routes'][0,0]=(int(a['routes'][0,0])+1)%4;np.savez(p,**a)
   with self.assertRaises(ValueError):replay_run(base/'runs/LF-s0')
if __name__=='__main__':unittest.main()
