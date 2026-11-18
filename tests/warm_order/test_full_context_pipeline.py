import json,tempfile,unittest
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from efficient_llm.warm_order.model import Config,Model as Dense
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.full_context_prepare import prepare,NAMES
from efficient_llm.warm_order.full_context_eval import run
from efficient_llm.warm_order.full_context_train import train
from efficient_llm.warm_order.full_context_report import report,replay_run
from efficient_llm.warm_order.route_gradient_eval import run as dense_control
from scripts.verify_full_context_export import verify

class PipelineTests(unittest.TestCase):
 def test_six_fits_reuse_two_dense_resume_pipeline_backup_and_tamper(self):
  torch.set_num_threads(1);torch.manual_seed(67);c=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=37,length=12,prefix=3)
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);stems={};hashes={}
   for domain in ('fit','dclm','fwe'):
    stem=str(r/domain);stems[domain]=stem;np.random.default_rng(3).integers(0,37,(32,12),dtype=np.uint32).tofile(stem+'.bin');st=np.zeros((32,12),dtype='u1');st[:,0]=1;st.tofile(stem+'.starts');hashes.update({stem+e:sha(stem+e) for e in ('.bin','.starts')})
   old=r/'old';old.mkdir()
   for s in (0,1):np.save(old/f'train-s{s}.npy',np.random.default_rng(s).permutation(8))
   (old/'complete.json').write_text(json.dumps(dict(files={p.name:sha(p) for p in old.glob('*.npy')})));hashes[str(old/'complete.json')]=sha(old/'complete.json')
   np.savez(r/'eval.npz',fit=np.arange(8,16),valid=np.arange(16,20),dclm=np.arange(4),fwe=np.arange(4));hashes[str(r/'eval.npz')]=sha(r/'eval.npz')
   pp=r/'parent.json';pp.write_text(json.dumps(dict(mechanism_all_seeds_domains=False,continuation_all_seeds_domains=False)));hashes[str(pp)]=sha(pp)
   ev=dict(stems=stems,indices=str(r/'eval.npz'));plan=dict(parent_report=str(pp),hashes=hashes,previous_inputs=str(old/'complete.json'),training={},weights={},window=5,stride=2,eval=ev,reused={})
   for seed in (0,1):
    m=Dense(c);p=r/f'dense-parent-s{seed}.pt';torch.save(dict(config=asdict(c),model=m.state_dict(),arm='D6C',window=5,stride=2,seed=seed),p);plan['weights'][str(seed)]=str(p);hashes[str(p)]=sha(p)
    tc=dict(model=asdict(c),steps=4,batch=2,lr=.0003,warmup=1,save_every=2,log_every=2,threads=1,data=stems['fit'],indices=str(old/f'train-s{seed}.npy'),seed=seed,window=5,stride=2,initial_weights=str(p),arm='D6C');plan['training'][str(seed)]=dict(tc)
    tc['expected_inputs']={p:sha(p) for p in (tc['data']+'.bin',tc['data']+'.starts',tc['indices'],tc['initial_weights'])}
    dest=r/f'prior/D6C-s{seed}';dense_control(dict(train=tc,eval=ev,hashes=dict(hashes)),dest,'cpu')
    plan['reused'][f'D6C-s{seed}']={}
    for domain in ('dclm','fwe'):
     p=dest/domain/'summary.npz';plan['reused'][f'D6C-s{seed}'][domain]=dict(path=str(p),sha256=sha(p));hashes[str(p)]=sha(p)
   base=r/'study';prepare(plan,base/'inputs');(base/'execution').mkdir()
   for p in ('source.json','admission.json'):(base/'execution'/p).write_text('{}')
   for name in NAMES:run(json.loads((base/f'inputs/{name}.json').read_text()),base/'runs'/name,'cpu')
   cfg=json.loads((base/'inputs/S-s0.json').read_text())['train'];train(cfg,r/'continuous','cpu');a=torch.load(base/'runs/S-s0/train/fullstate.pt',weights_only=False);b=torch.load(r/'continuous/fullstate.pt',weights_only=False)
   def equal(x,y):
    if torch.is_tensor(x):torch.testing.assert_close(x,y,rtol=0,atol=0)
    elif isinstance(x,dict):
     self.assertEqual(x.keys(),y.keys())
     for k in x:equal(x[k],y[k])
    elif isinstance(x,(tuple,list)):
     for i,j in zip(x,y):equal(i,j)
    else:self.assertEqual(x,y)
   equal(a['model'],b['model']);equal(a['optimizer'],b['optimizer']);equal(a['torch_rng'],b['torch_rng'])
   before={p:sha(p) for p in plan['hashes']};result=report(base,r/'backup');self.assertEqual(before,{p:sha(p) for p in before});self.assertEqual(len(result['rows']),16);self.assertEqual(len(json.loads((base/'backup.json').read_text())['node_copies']),13);self.assertEqual(verify(base/'results.tar.gz',base/'backup.json',r/'verified')['status'],'PASS')
   p=base/'runs/S-s0/dclm/summary.npz';a=dict(np.load(p));a['routes'][0,0]=(int(a['routes'][0,0])+1)%4;np.savez(p,**a)
   with self.assertRaises(ValueError):replay_run(base/'runs/S-s0')
if __name__=='__main__':unittest.main()
