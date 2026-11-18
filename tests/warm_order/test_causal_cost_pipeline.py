import json,tempfile,unittest
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from efficient_llm.warm_order.model import Config,Model as Dense
from efficient_llm.warm_order.window_film import Model as Film
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.causal_cost_prepare import prepare
from efficient_llm.warm_order.causal_cost_report import NAMES,report
from efficient_llm.warm_order.causal_cost_run import cache_job,fit_job,evaluate_job
from scripts.verify_causal_cost_export import verify


class PipelineTest(unittest.TestCase):
    def test_four_frozen_parents_eight_caches_selection_evals_backup_replay(self):
        torch.set_num_threads(1);torch.manual_seed(7)
        c=Config(d=8,heads=2,kv_heads=1,head_dim=4,ff=16,vocab=17,length=9,prefix=2)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stems={}
            for name in ('fit','dclm','fwe'):
                stem=root/name;stems[name]=str(stem);np.random.default_rng(19).integers(0,17,(20,9),dtype='<u4').tofile(str(stem)+'.bin');a=np.zeros((20,9),dtype='u1');a[:,0]=1;a.tofile(str(stem)+'.starts')
            np.savez(root/'ids.npz',fit=np.arange(8),valid=np.arange(8,16),dclm=np.arange(3),fwe=np.arange(3))
            (root/'trigger.json').write_text(json.dumps(dict(LF_both_seeds_headroom_gt_005=dict(hold=True))))
            plan=dict(stems=stems,indices=str(root/'ids.npz'),parent_report=str(root/'trigger.json'),fit_n=8,valid_start=2,valid_stop=8,length=9,epochs=3,hidden=12,weights={a:{} for a in ('GF','LF','D6C')})
            for arm in ('GF','LF','D6C'):
                for seed in (0,1):
                    m=Dense(c) if arm=='D6C' else Film(c,'WG' if arm=='GF' else 'WL',4,2,seed)
                    if arm!='D6C':
                        with torch.no_grad():m.router.weight.normal_(std=.2)
                    p=root/f'{arm}-{seed}.pt';torch.save(dict(config=asdict(c),arm=arm,seed=seed,window=4,stride=2,model=m.state_dict()),p);plan['weights'][arm][str(seed)]=str(p)
            plan['hashes']={str(p):sha(p) for p in root.iterdir() if p.is_file()};before=dict(plan['hashes']);base=root/'study';prepare(plan,base/'inputs')
            (base/'execution').mkdir()
            for name in ('source.json','admission.json'):(base/'execution'/name).write_text('{}')
            for name in NAMES:
                config=json.loads((base/'inputs'/f'{name}.json').read_text());run=base/'runs'/name
                for part in (0,1):cache_job(config,run/f'cache-{part}',part,'cpu')
                fit_job(config,run)
                for domain in ('dclm','fwe'):evaluate_job(config,run,domain,'cpu')
            result=report(base,root/'backup');self.assertEqual(len(result['rows']),8)
            self.assertEqual(len(json.loads((base/'backup.json').read_text())['node_copies']),9)
            self.assertEqual(verify(base/'results.tar.gz',base/'backup.json',root/'verified')['status'],'PASS')
            self.assertEqual(before,{p:sha(p) for p in before})
            a=dict(np.load(base/'runs/GF-s0/eval/dclm/summary.npz'));a['learned'][0]+=.1;np.savez(base/'runs/GF-s0/eval/dclm/summary.npz',**a)
            with self.assertRaises(ValueError):report(base,root/'bad-backup')


if __name__=='__main__':unittest.main()
