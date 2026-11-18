import json,tempfile,unittest
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from efficient_llm.warm_order.model import Config,Model
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.refinement_train import train
from efficient_llm.warm_order.refinement_eval import evaluate
from efficient_llm.warm_order.strength_refinement import prepare,run,NAMES
from efficient_llm.warm_order.strength_report import report,calculate
from scripts.verify_strength_refinement_export import verify

class StrengthTests(unittest.TestCase):
 def test_real_pipeline_resume_and_final_selection_independence(self):
  torch.set_num_threads(1);torch.manual_seed(193)
  c=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=37,prefix=3,length=12)
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);prior=root/'prior';prior.mkdir();stems={};hashes={}
   for domain in ('fit','dclm','fwe'):
    s=str(root/domain);stems[domain]=s;np.random.default_rng(9).integers(0,37,(24,12),dtype=np.uint32).tofile(s+'.bin');starts=np.zeros((24,12),dtype='u1');starts[:,0]=1;starts.tofile(s+'.starts')
    hashes.update({s+ext:sha(s+ext) for ext in ('.bin','.starts')})
   np.save(root/'train.npy',np.arange(8));np.savez(root/'eval.npz',fit=np.arange(8,12),valid=np.arange(12,14),dclm=np.arange(4),fwe=np.arange(4))
   ix=str(root/'eval.npz');ev=dict(stems=stems,indices=ix,hashes={**hashes,ix:sha(ix)},final_fresh_documents=dict(dclm=False,fwe=False));plan=dict(hashes={},parents={})
   for seed in (0,1):
    initial=root/f'initial{seed}.pt';torch.save(dict(config=asdict(c),model=Model(c).state_dict()),initial)
    for arm in ('C6','F7'):
     name=f'{arm}-s{seed}';tc=dict(model=asdict(c),data=stems['fit'],indices=str(root/'train.npy'),initial_weights=str(initial),steps=4,batch=2,seed=seed,arm=arm,lr=1e-4,warmup=1,log_every=1,save_every=2)
     tc['expected_inputs']={p:sha(p) for p in (tc['data']+'.bin',tc['data']+'.starts',tc['indices'],tc['initial_weights'])}
     train(tc,prior/'runs'/name,'cpu');evaluate(ev,prior/'runs'/name,prior/'eval'/name,'cpu',head_epochs=1)
    f=prior/f'eval/F7-s{seed}';plan['parents'][str(seed)]=dict(recipe=str(prior/f'runs/F7-s{seed}/recipe.json'),eval_config=str(f/'config.json'),prior=str(f),c6={dm:str(prior/f'eval/C6-s{seed}/{dm}/summary.npz') for dm in ('dclm','fwe')})
   plan['hashes']={str(p):sha(p) for p in prior.rglob('*') if p.is_file()}
   base=root/'new';prepare(plan,base/'inputs');(base/'execution').mkdir()
   for p in ('source.json','admission.json'):(base/'execution'/p).write_text('{}')
   for name in NAMES:run(json.loads((base/f'inputs/{name}.json').read_text()),base/'runs'/name,'cpu')
   # Compare exact fullstate continuation against an uninterrupted higher-LR run.
   cfg=json.loads((base/'inputs/H10-s0.json').read_text())['train'];train(cfg,root/'full','cpu')
   a=torch.load(root/'full/fullstate.pt',weights_only=False);b=torch.load(base/'runs/H10-s0/train/fullstate.pt',weights_only=False)
   for k,v in a['model'].items():torch.testing.assert_close(v,b['model'][k],atol=0,rtol=0)
   self.assertEqual(a['step'],b['step']);self.assertEqual(a['optimizer']['param_groups'],b['optimizer']['param_groups'])
   for k,st in a['optimizer']['state'].items():
    for key,v in st.items():torch.testing.assert_close(v,b['optimizer']['state'][k][key],atol=0,rtol=0)
   result=report(base,root/'backup');self.assertEqual(len(json.loads((base/'backup.json').read_text())['node_copies']),9)
   self.assertEqual(verify(base/'results.tar.gz',base/'backup.json',root/'verified',target_count=9)['status'],'PASS')
   arrays={(name,dm):dict(np.load(base/'runs'/name/dm/'summary.npz')) for name in NAMES for dm in ('valid','dclm','fwe')}
   selected=calculate(arrays)['selected']
   for (name,dm),v in arrays.items():
    if dm!='valid':v['refine']=v['refine']-100
   self.assertEqual(calculate(arrays)['selected'],selected)
   # Reuse and mutation must fail closed, without overwriting prior experiments.
   with self.assertRaises(ValueError):prepare(plan,base/'inputs')
   p=base/'runs/H3-s0/dclm/costs.npy';a=np.load(p);a[0,0,0]+=1;np.save(p,a)
   with self.assertRaises(ValueError):report(base,root/'bad-backup')

if __name__=='__main__':unittest.main()
