import copy,json,tempfile,unittest
from pathlib import Path
from dataclasses import asdict
from unittest.mock import patch
import numpy as np
import torch
from efficient_llm.warm_order.model import Config,Model as Dense
from efficient_llm.warm_order.window_film import Model as Film
from efficient_llm.warm_order.window_cf import Intervention,enumerate_costs,run,PATH
from efficient_llm.warm_order.window_cf_prepare import prepare,NAMES
from efficient_llm.warm_order.window_cf_report import report,replay_run
from efficient_llm.warm_order.data import Rows,sha
from scripts.verify_window_cf_export import verify

class CounterfactualTests(unittest.TestCase):
 def test_factual_identity_intervention_causality_and_cost_enumeration(self):
  torch.set_num_threads(1);torch.manual_seed(31);c=Config(d=8,heads=2,kv_heads=1,head_dim=4,ff=16,vocab=17,length=9,prefix=2);m=Intervention(c,'WL',4,2,0).eval();x=torch.randint(17,(3,9))
  with torch.no_grad():m.router.weight.normal_(std=.2);m.router.bias.normal_(std=.1);m.film_gamma.normal_(std=.1);m.film_beta.normal_(std=.1)
  base=m.hidden(x);routes=[r.clone() for r in m.last_routes]
  for mode in ('hold','rescore'):
   for depth,rr in enumerate(routes):
    for j in range(rr.shape[1]):
     m.intervention=(depth,j,rr[:,j],mode);torch.testing.assert_close(m.hidden(x),base,rtol=0,atol=0)
   m.intervention=(2,1,3,mode);a=m.hidden(x)
   for cut in (1,2,4,8):
    z=x.clone();z[:,cut:]=(z[:,cut:]+3)%17;torch.testing.assert_close(a[:,:cut],m.hidden(z)[:,:cut],rtol=0,atol=0)
  m.intervention=None
  with tempfile.TemporaryDirectory() as d:
   stem=str(Path(d)/'data');x.numpy().astype(np.uint32).tofile(stem+'.bin');np.zeros((3,9),dtype='u1').tofile(stem+'.starts');rows=Rows(stem,9);a=enumerate_costs(m,rows,np.arange(3),'cpu','rescore')
   for col,(depth,j,k) in enumerate(a['actions']):
    m.intervention=(int(depth),int(j),int(k),'rescore');expected=np.array([float(m.nll(x[i:i+1],PATH).detach()[0]) for i in range(3)]);np.testing.assert_allclose(a['costs'][:,col],expected,rtol=1e-5,atol=1e-6)
   m.intervention=None

 def test_eight_frozen_jobs_report_backup_replay_no_parent_mutation(self):
  torch.set_num_threads(1);torch.manual_seed(9);c=Config(d=8,heads=2,kv_heads=1,head_dim=4,ff=16,vocab=17,length=9,prefix=2)
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);stem=str(root/'data');np.random.default_rng(3).integers(0,17,(12,9),dtype=np.uint32).tofile(stem+'.bin');np.zeros((12,9),dtype='u1').tofile(stem+'.starts');np.savez(root/'ids.npz',valid=np.arange(3));(root/'parent-report.json').write_text(json.dumps(dict(continuation_all_seeds_domains=False)))
   plan=dict(sequences=3,data=stem,indices=str(root/'ids.npz'),parent_report=str(root/'parent-report.json'),weights={a:{} for a in ('GF','LF','D6C')},hashes={str(p):sha(p) for p in root.iterdir() if p.is_file()})
   for arm in ('GF','LF','D6C'):
    for seed in (0,1):
     m=Dense(c) if arm=='D6C' else Film(c,'WG' if arm=='GF' else 'WL',4,2,seed,film=True)
     if arm!='D6C':
      with torch.no_grad():m.router.weight.normal_(std=.1)
     p=root/f'{arm}-{seed}.pt';torch.save(dict(config=asdict(c),arm=arm,seed=seed,window=4,stride=2,model=m.state_dict()),p);plan['weights'][arm][str(seed)]=str(p);plan['hashes'][str(p)]=sha(p)
   before=dict(plan['hashes']);base=root/'study';prepare(plan,base/'inputs');(base/'execution').mkdir()
   for f in ('source.json','admission.json'):(base/'execution'/f).write_text('{}')
   with patch('efficient_llm.warm_order.window_film_train.train',side_effect=AssertionError('No training allowed')):
    for n in NAMES:run(json.loads((base/'inputs'/(n+'.json')).read_text()),base/'runs'/n,'cpu')
   result=report(base,root/'backup');self.assertEqual(len(result['rows']),8);self.assertEqual(len(json.loads((base/'backup.json').read_text())['node_copies']),1);self.assertEqual(verify(base/'results.tar.gz',base/'backup.json',root/'verified')['status'],'PASS')
   self.assertEqual(before,{p:sha(p) for p in before});p=base/'runs/LF-s0-hold/costs.npz';a=dict(np.load(p));a['costs'][0,0]+=1;np.savez(p,**a)
   with self.assertRaises(ValueError):replay_run(p.parent)
if __name__=='__main__':unittest.main()
