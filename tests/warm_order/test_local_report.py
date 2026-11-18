import json,tempfile,unittest
from pathlib import Path
import numpy as np
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.local_report import report,NAMES


class LocalReportTests(unittest.TestCase):
    def test_eight_runs_replay_backup_and_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);base=root/'run'
            def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))
            write(base/'execution/source.json',{});write(base/'inputs/complete.json',{'flags':['fixture']})
            rng=np.random.default_rng(15)
            for name in NAMES:
                arm,seed=name.split('-s');run=base/'runs'/name;ev=base/'eval'/name
                run.mkdir(parents=True);ev.mkdir(parents=True)
                for f in ('weights.pt','fullstate.pt'):(run/f).write_bytes(b'fixture, not a checkpoint')
                done=dict(step=6,weights_sha256=sha(run/'weights.pt'),fullstate_sha256=sha(run/'fullstate.pt'))
                for f in ('complete.json','recipe.json','checkpoint.json'):write(run/f,done)
                constant=0;domains={}
                if arm!='C':
                    fit=rng.random((8,4),dtype=np.float32);constant=int(fit.mean(0).argmin())
                    np.savez(ev/'fit_costs.npz',costs=fit,ids=np.arange(8))
                for domain in ('dclm','fwe'):
                    cost=rng.random((8,1 if arm=='C' else 4),dtype=np.float32)
                    choices=np.arange(8)%cost.shape[1];metrics={'canonical':cost[:,0]}
                    if arm!='C':metrics.update(constant=cost[:,constant],uniform=cost.mean(1),oracle=cost.min(1),mixture=cost@np.array([.79,.07,.07,.07],dtype=np.float32))
                    if arm in ('CF','PG'):metrics['learned']=cost[np.arange(8),choices]
                    np.savez(ev/f'{domain}_costs.npz',costs=cost,choices=choices,ids=np.arange(8),clusters=np.arange(8)//2,**metrics)
                    domains[domain]={'metrics':{k:float(v.mean()) for k,v in metrics.items()}}
                write(ev/'report.json',dict(arm=arm,seed=int(seed),weights_sha256=done['weights_sha256'],constant=constant,domains=domains))
                write(ev/'complete.json',dict(files={p.name:sha(p) for p in ev.iterdir() if p.is_file()}))
            result=report(base,root/'backup',steps=6)
            self.assertEqual(len(result['rows']),8);self.assertEqual(len(result['comparisons']),20)
            proof=json.loads((base/'backup.json').read_text());self.assertEqual(len(proof['node_checkpoint_copies']),40)
            self.assertFalse(any(p.endswith('.pt') for p in proof['files']))
            for p,want in proof['node_checkpoint_copies'].items():self.assertEqual(sha(p),want)
            (base/'eval/PG-s1/dclm_costs.npz').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'Evaluation changed'):report(base,root/'backup2',steps=6)
            self.assertFalse((root/'backup2').exists())


if __name__=='__main__':unittest.main()
