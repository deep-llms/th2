import json,tempfile,unittest
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from efficient_llm.warm_order.model import Config,Model
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.soft_route_prepare import prepare,NAMES
from efficient_llm.warm_order.soft_route_eval import run
from efficient_llm.warm_order.soft_route_report import report,replay_run
from scripts.verify_soft_route_export import verify

class PipelineTests(unittest.TestCase):
    def test_six_fits_reservation_report_backup_replay_and_tamper(self):
        torch.set_num_threads(1);torch.manual_seed(67)
        c=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=37,length=12,prefix=3)
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);stems={};hashes={}
            for domain in ('fit','dclm','fwe'):
                stem=str(r/domain);stems[domain]=stem
                np.random.default_rng(3).integers(0,37,(48,12),dtype=np.uint32).tofile(stem+'.bin')
                starts=np.zeros((48,12),dtype='u1');starts[:,0]=1;starts.tofile(stem+'.starts')
                hashes.update({stem+e:sha(stem+e) for e in ('.bin','.starts')})
            old=r/'old';old.mkdir()
            for s in (0,1):np.save(old/f'train-s{s}.npy',np.arange(8))
            (old/'complete.json').write_text(json.dumps(dict(files={p.name:sha(p) for p in old.glob('*.npy')})))
            hashes[str(old/'complete.json')]=sha(old/'complete.json')
            np.savez(r/'eval.npz',fit=np.arange(8,12),valid=np.arange(12,16),dclm=np.arange(4),fwe=np.arange(4))
            hashes[str(r/'eval.npz')]=sha(r/'eval.npz')
            pp=r/'parent.json';pp.write_text(json.dumps(dict(curriculum_all_seeds_domains=False,continuation_all_seeds_domains=False)));hashes[str(pp)]=sha(pp)
            ev=dict(stems=stems,indices=str(r/'eval.npz'))
            plan=dict(parent_report=str(pp),parent_inputs=str(old/'complete.json'),hashes=hashes,training={},weights={},parent_metadata={},eval=ev)
            for s in (0,1):
                parent_dir=r/f'parent-s{s}';parent_dir.mkdir();p=parent_dir/'weights.pt'
                torch.save(dict(config=asdict(c),model=Model(c).state_dict()),p)
                (parent_dir/'complete.json').write_text(json.dumps(dict(arm='D',seed=s,step=80,weights_sha256=sha(p))))
                (parent_dir/'recipe.json').write_text(json.dumps(dict(config=dict(arm='D',seed=s,steps=80,model=asdict(c)))))
                plan['parent_metadata'][str(s)]={str(parent_dir/f):sha(parent_dir/f) for f in ('complete.json','recipe.json')}
                hashes.update(plan['parent_metadata'][str(s)])
                hashes[str(p)]=sha(p);plan['weights'][str(s)]=str(p)
                plan['training'][str(s)]=dict(model=asdict(c),batch=2,steps=4,lr=.0001,router_lr=.001,warmup=1,save_every=2,log_every=2,threads=1,data=stems['fit'],canonical_probability=.999,parent_steps=80)
            base=r/'study';prepare(plan,base/'inputs');(base/'execution').mkdir()
            for p in ('source.json','admission.json'):(base/'execution'/p).write_text('{}')
            reservation=json.loads((base/'inputs/complete.json').read_text())['reservation']
            self.assertTrue(all(v['parent_row_overlap']==v['reserved_doc_overlap']==0 for v in reservation.values()))
            for name in NAMES:run(json.loads((base/f'inputs/{name}.json').read_text()),base/'runs'/name,'cpu')
            before={p:sha(p) for p in hashes};result=report(base,r/'backup')
            self.assertEqual(before,{p:sha(p) for p in hashes});self.assertEqual(len(result['rows']),24)
            self.assertEqual(len(json.loads((base/'backup.json').read_text())['node_copies']),13)
            self.assertEqual(verify(base/'results.tar.gz',base/'backup.json',r/'verified')['status'],'PASS')
            p=base/'runs/S-s0/dclm/summary.npz';a=dict(np.load(p));a['soft_counts'][0,0,0]+=1;np.savez(p,**a)
            with self.assertRaises(ValueError):replay_run(base/'runs/S-s0')

if __name__=='__main__':unittest.main()
