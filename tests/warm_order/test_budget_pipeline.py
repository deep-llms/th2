import unittest,tempfile,json
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from efficient_llm.warm_order.model import Config,Model
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.refinement_train import train
from efficient_llm.warm_order.refinement_eval import evaluate as parent_eval
from efficient_llm.warm_order.budget_eval import evaluate
from efficient_llm.warm_order.budget_report import NAMES,report
from scripts.verify_budget_refinement_export import verify

class PipelineTests(unittest.TestCase):
 def test_real_frozen_heads_dispatch_report_backup_export(self):
  torch.set_num_threads(1);torch.manual_seed(4);c=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=37,prefix=3,length=12)
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);parent=root/'parent';base=root/'budget';(base/'inputs').mkdir(parents=True);(base/'execution').mkdir();parent.mkdir()
   for p in ('inputs/complete.json','execution/source.json','execution/admission.json'):(base/p).write_text('{}')
   stems={};hashes={}
   for domain in ('fit','dclm','fwe'):
    s=str(root/domain);stems[domain]=s;np.random.default_rng(9).integers(0,37,(20,12),dtype=np.uint32).tofile(s+'.bin');starts=np.zeros((20,12),dtype='u1');starts[:,0]=1;starts.tofile(s+'.starts');hashes.update({s+ext:sha(s+ext) for ext in ('.bin','.starts')})
   np.save(root/'train.npy',np.arange(8));np.savez(root/'eval.npz',fit=np.arange(8,12),valid=np.arange(12,14),dclm=np.arange(4),fwe=np.arange(4));ix=str(root/'eval.npz');ec=dict(stems=stems,indices=ix,hashes={**hashes,ix:sha(ix)},final_fresh_documents=dict(dclm=True,fwe=True))
   for name in ['C6-s0','C6-s1',*NAMES]:
    arm=name.split('-')[0];seed=int(name[-1]);init=root/(name+'.pt');torch.save(dict(config=asdict(c),model=Model(c).state_dict()),init)
    tc=dict(model=asdict(c),data=stems['fit'],indices=str(root/'train.npy'),initial_weights=str(init),steps=4,batch=2,seed=seed,arm=arm,lr=.001,warmup=1,log_every=2,save_every=2)
    tc['expected_inputs']={p:sha(p) for p in (tc['data']+'.bin',tc['data']+'.starts',tc['indices'],tc['initial_weights'])};train(tc,parent/'runs'/name,'cpu');parent_eval(ec,parent/'runs'/name,parent/'eval'/name,'cpu',head_epochs=2)
   for name in NAMES:
    prior=parent/'eval'/name;weight=parent/'runs'/name/'weights.pt';baseline={dm:str(parent/'eval'/('C6-s'+name[-1])/dm/'summary.npz') for dm in ('dclm','fwe')};files=[weight,*prior.rglob('*'),*[Path(v) for v in baseline.values()]]
    cfg=dict(name=name,seed=int(name[-1]),prior=str(prior),weights=str(weight),c6=baseline,hashes={**hashes,**{str(p):sha(p) for p in files if p.is_file()}})
    evaluate(cfg,base/'runs'/name,'cpu',epochs=2)
   result=report(base,root/'backup');self.assertEqual(result['node_raw_replay'],'PASS');self.assertEqual(len(result['comparisons']),72)
   self.assertEqual(verify(base/'results.tar.gz',base/'backup.json',root/'export',target_count=9)['status'],'PASS')
if __name__=='__main__':unittest.main()
