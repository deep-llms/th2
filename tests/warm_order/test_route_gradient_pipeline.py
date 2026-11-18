import json,tempfile,unittest,copy
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from efficient_llm.warm_order.model import Config,Model as Dense
from efficient_llm.warm_order.window_film import Model as Film
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.route_gradient_prepare import prepare,NAMES
from efficient_llm.warm_order.route_gradient_eval import run
from efficient_llm.warm_order.route_gradient_train import train
from efficient_llm.warm_order.route_gradient_report import report,replay_run
from scripts.verify_route_gradient_export import verify

class PipelineTests(unittest.TestCase):
 def test_eight_fits_resume_reservation_eval_report_backup_export_tamper(self):
  torch.set_num_threads(1);torch.manual_seed(42);c=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=37,length=12,prefix=3)
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);stems={};hashes={}
   for domain in ('fit','dclm','fwe'):
    stem=str(r/domain);stems[domain]=stem;np.random.default_rng(9).integers(0,37,(64,12),dtype=np.uint32).tofile(stem+'.bin');st=np.zeros((64,12),dtype='u1');st[:,0]=1;st.tofile(stem+'.starts');hashes.update({stem+e:sha(stem+e) for e in ('.bin','.starts')})
   old=r/'old';old.mkdir();np.save(old/'train-s0.npy',np.arange(8));(old/'complete.json').write_text(json.dumps(dict(files={'train-s0.npy':sha(old/'train-s0.npy')})));hashes[str(old/'complete.json')]=sha(old/'complete.json')
   np.savez(r/'eval.npz',fit=np.arange(8,16),valid=np.arange(16,20),dclm=np.arange(4),fwe=np.arange(4));hashes[str(r/'eval.npz')]=sha(r/'eval.npz')
   pp=r/'parent.json';pp.write_text(json.dumps(dict(criterion={'GF':False,'LF':False})));hashes[str(pp)]=sha(pp)
   plan=dict(parent_report=str(pp),hashes=hashes,training=dict(model=asdict(c),steps=4,batch=2,lr=.0003,warmup=1,save_every=2,log_every=2,threads=1,data=stems['fit']),weights={k:{} for k in ('GF','LF','D6C')},prior_input_manifests=[str(old/'complete.json')],window=5,stride=2,eval=dict(stems=stems,indices=str(r/'eval.npz')))
   for seed in (0,1):
    for arm in ('GF','LF','D6C'):
     m=Dense(c) if arm=='D6C' else Film(c,'WG' if arm=='GF' else 'WL',5,2,seed)
     p=r/f'{arm}-s{seed}.pt';torch.save(dict(config=asdict(c),model=m.state_dict(),arm=arm,window=5,stride=2,seed=seed),p);plan['weights'][arm][str(seed)]=str(p);hashes[str(p)]=sha(p)
   pp.write_text(json.dumps(dict(criterion={'GF':True,'LF':False})));hashes[str(pp)]=sha(pp)
   with self.assertRaises(ValueError):prepare(plan,r/'invalid')
   self.assertFalse((r/'invalid').exists());pp.write_text(json.dumps(dict(criterion={'GF':False,'LF':False})));hashes[str(pp)]=sha(pp)
   base=r/'study';prepare(plan,base/'inputs');(base/'execution').mkdir()
   for p in ('source.json','admission.json'):(base/'execution'/p).write_text('{}')
   for seed in (0,1):
    ids=np.load(base/f'inputs/train-s{seed}.npy');self.assertEqual(len(ids),8);self.assertEqual(np.intersect1d(ids,np.arange(20)).size,0)
   np.testing.assert_array_equal(np.sort(np.load(base/'inputs/train-s0.npy')),np.sort(np.load(base/'inputs/train-s1.npy')))
   for name in NAMES:run(json.loads((base/f'inputs/{name}.json').read_text()),base/'runs'/name,'cpu')
   # Full RNG/optimizer/schedule continuation equals uninterrupted actual trainer.
   for arm in ('HG','SG'):
    cfg=json.loads((base/f'inputs/{arm}-s0.json').read_text())['train'];train(cfg,r/(arm+'-continuous'),'cpu')
    a=torch.load(base/f'runs/{arm}-s0/train/fullstate.pt',weights_only=False);b=torch.load(r/(arm+'-continuous')/'fullstate.pt',weights_only=False)
    def equal(x,y):
     if torch.is_tensor(x):torch.testing.assert_close(x,y,rtol=0,atol=0)
     elif isinstance(x,dict):
      self.assertEqual(x.keys(),y.keys())
      for k in x:equal(x[k],y[k])
     elif isinstance(x,(list,tuple)):
      for i,j in zip(x,y):equal(i,j)
     else:self.assertEqual(x,y)
    equal(a['model'],b['model']);equal(a['optimizer'],b['optimizer']);equal(a['torch_rng'],b['torch_rng']);self.assertEqual(a['step'],4)
   result=report(base,r/'backup');self.assertEqual(result['node_raw_replay'],'PASS');self.assertEqual(len(json.loads((base/'backup.json').read_text())['node_copies']),17);self.assertEqual(verify(base/'results.tar.gz',base/'backup.json',r/'verified')['status'],'PASS')
   self.assertEqual(result['mechanism_all_seeds_domains'],all(result['comparisons'][f'SG-s{s}/{d}/minus-HG']['mean']<=-.005 for s in (0,1) for d in ('dclm','fwe')))
   p=base/'runs/SG-s0/dclm/summary.npz';a=dict(np.load(p));a['routes'][0,0]=(int(a['routes'][0,0])+1)%4;np.savez(p,**a)
   with self.assertRaises(ValueError):replay_run(base/'runs/SG-s0')
if __name__=='__main__':unittest.main()
