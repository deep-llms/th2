import json,tempfile,unittest,tarfile
from pathlib import Path
import numpy as np
from efficient_llm.warm_order.train import write_json
from efficient_llm.warm_order.data import sha
from scripts.warm_order_queue import report
class Report(unittest.TestCase):
    def test_replay_backup_and_no_large_archive(self):
        with tempfile.TemporaryDirectory() as d:
            b=Path(d)/'base';b.mkdir();(b/'execution').mkdir();(b/'inputs').mkdir()
            for f in ['gpu-probe.json','execution/source.json','inputs/complete.json']:write_json(b/f,{})
            for arm in ('D','R','W'):
                for seed in (0,1):
                    name=f'{arm}-s{seed}';run=b/'runs'/name;ev=b/'eval'/name;run.mkdir(parents=True);ev.mkdir(parents=True)
                    for f in ['weights.pt','fullstate.pt']:(run/f).write_bytes(name.encode())
                    for f in ['recipe.json','checkpoint.json']:write_json(run/f,{})
                    write_json(run/'complete.json',dict(step=12000,weights_sha256=sha(run/'weights.pt'),fullstate_sha256=sha(run/'fullstate.pt')))
                    c=np.tile(np.arange(1,9,dtype='f'),(16,1)) if arm!='D' else np.ones((16,1),dtype='f')
                    m=dict(canonical=c[:,0])
                    if arm!='D':m.update(uniform=c.mean(1),constant=c[:,0],learned=c[:,0],oracle=c.min(1))
                    np.savez(ev/'test_costs.npz',costs=c,choices=np.zeros(16,dtype='i'),ids=np.arange(16),clusters=np.arange(16)//2,**m)
                    write_json(ev/'report.json',dict(arm=arm,seed=seed,constant=0,metrics={k:float(v.mean()) for k,v in m.items()}))
                    write_json(ev/'complete.json',dict(files={p.name:sha(p) for p in ev.iterdir()}))
            report(b,Path(d)/'backup')
            proof=json.loads((b/'backup.json').read_text())
            for path,want in proof['node_checkpoint_copies'].items():self.assertEqual(sha(path),want)
            with tarfile.open(b/'tier1.tar.gz') as t:
                self.assertFalse(any(p.endswith(('weights.pt','fullstate.pt')) for p in t.getnames()))
            self.assertEqual(json.loads((b/'REPORT.json').read_text())['mean_over_seeds']['W']['learned'],1.)
if __name__=='__main__':unittest.main()
