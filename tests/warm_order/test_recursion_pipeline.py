import json,tempfile,unittest
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from efficient_llm.warm_order.model import Config
from efficient_llm.warm_order.train import train as dense_train
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.recursion_prepare import prepare,NAMES
from efficient_llm.warm_order.recursion_eval import run
from efficient_llm.warm_order.recursion_report import report,calculate,replay_run
from efficient_llm.warm_order.recursion_head import calibrate,choose
from scripts.verify_recursive_depth_export import verify

class PipelineTests(unittest.TestCase):
 def test_actual_six_fits_two_reused_eval_budget_report_export(self):
  torch.set_num_threads(1);torch.manual_seed(4);c=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=37,length=12,prefix=3)
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);stems={};hashes={}
   for domain in ('fit','dclm','fwe'):
    s=str(r/domain);stems[domain]=s;np.random.default_rng(9).integers(0,37,(40,12),dtype=np.uint32).tofile(s+'.bin');st=np.zeros((40,12),dtype='u1');st[:,0]=1;st.tofile(s+'.starts');hashes.update({s+e:sha(s+e) for e in ('.bin','.starts')})
   np.save(r/'train.npy',np.arange(8));np.savez(r/'eval.npz',fit=np.arange(8,16),valid=np.arange(16,20),dclm=np.arange(4),fwe=np.arange(4));hashes[str(r/'eval.npz')]=sha(r/'eval.npz')
   plan=dict(hashes=hashes,training={},weights={},router_epochs=2,eval=dict(stems=stems,indices=str(r/'eval.npz')))
   for seed in (0,1):
    tc=dict(model=asdict(c),arm='D',steps=4,batch=2,seed=seed,lr=.001,warmup=1,save_every=2,log_every=2,data=stems['fit'],indices=str(r/'train.npy'));tc['expected_inputs']={p:sha(p) for p in (stems['fit']+'.bin',stems['fit']+'.starts',tc['indices'])};plan['hashes'].update(tc['expected_inputs']);dense_train(tc,r/f'D6-s{seed}','cpu');plan['training'][str(seed)]=tc;w=str(r/f'D6-s{seed}/weights.pt');plan['weights'][str(seed)]=w;plan['hashes'][w]=sha(w)
   base=r/'study';prepare(plan,base/'inputs');(base/'execution').mkdir()
   for p in ('source.json','admission.json'):(base/'execution'/p).write_text('{}')
   for name in NAMES:run(json.loads((base/f'inputs/{name}.json').read_text()),base/'runs'/name,'cpu')
   result=report(base,r/'backup');self.assertEqual(result['node_raw_replay'],'PASS');self.assertEqual(len(json.loads((base/'backup.json').read_text())['node_copies']),15)
   self.assertEqual(verify(base/'results.tar.gz',base/'backup.json',r/'verified')['status'],'PASS')
   p=base/'runs/RV-s0/dclm/summary.npz';a=dict(np.load(p));a['choices'][0]=(a['choices'][0]+1)%3;np.savez(p,**a)
   with self.assertRaises(ValueError):replay_run(base/'runs/RV-s0')
   with self.assertRaises(ValueError):prepare(plan,base/'inputs')
 def test_budget_and_fixed_per_example_policy(self):
  rng=np.random.default_rng(1);scores=rng.normal(size=(100,3)).astype('f4');cost=rng.normal(size=(100,3)).astype('f4');p=calibrate(scores,cost,1.875);ch=choose(scores,p);self.assertLessEqual(float((ch+1).mean()),1.875)
  np.testing.assert_array_equal(ch,np.r_[choose(scores[:10],p),choose(scores[10:],p)])
  changed=scores.copy();changed[1:]*=100;self.assertEqual(ch[0],choose(changed,p)[0])
  p=calibrate(np.zeros((5,3)),np.ones((5,3)),1.0);self.assertTrue((choose(np.zeros((5,3)),p)==0).all())
if __name__=='__main__':unittest.main()
