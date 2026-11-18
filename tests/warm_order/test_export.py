import hashlib,json,tarfile,tempfile,unittest
from pathlib import Path
import numpy as np
from scripts.verify_warm_order_export import sha,verify
class Export(unittest.TestCase):
    def test_export_replay_and_hash_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'src';src.mkdir();reports=[]
            for arm in ('D','R','W'):
                for seed in (0,1):
                    name=f'{arm}-s{seed}';ev=src/f'eval/{name}';run=src/f'runs/{name}';ev.mkdir(parents=True);run.mkdir(parents=True)
                    cost=np.ones((8192,1 if arm=='D' else 8),dtype='f')*4
                    if arm!='D':cost+=np.arange(8,dtype='f')*.01
                    metrics=dict(canonical=cost[:,0])
                    if arm!='D':metrics.update(uniform=cost.mean(1),constant=cost[:,0],learned=cost[:,0],oracle=cost.min(1))
                    np.savez(ev/'test_costs.npz',costs=cost,choices=np.zeros(8192,dtype='i'),ids=np.arange(8192),clusters=np.arange(8192)//4,**metrics)
                    if arm!='D':np.savez(ev/'fit_dev_costs.npz',fit=cost[:64],dev=cost[:64])
                    row=dict(arm=arm,seed=seed,constant=0,weights_sha256='a'*64,metrics={k:float(v.mean()) for k,v in metrics.items()});reports.append(row)
                    (ev/'report.json').write_text(json.dumps(row));(run/'complete.json').write_text(json.dumps(dict(step=12000,input_tokens=196608000,scored_tokens=184320000,weights_sha256='a'*64)))
            (src/'REPORT.json').write_text(json.dumps(dict(rows=reports,comparisons={f'W_learned_minus_D_s{s}':dict(mean=0.) for s in (0,1)})))
            files={str(p.relative_to(src)):sha(p) for p in src.rglob('*') if p.is_file()};archive=root/'tier1.tar.gz'
            with tarfile.open(archive,'w:gz') as t:
                for p in files:t.add(src/p,arcname=p)
            proof=root/'backup.json';proof.write_text(json.dumps(dict(bytes=archive.stat().st_size,archive_sha256=sha(archive),files=files)))
            result=verify(archive,proof,root/'verified');self.assertEqual(result['run_count'],6)
            content=json.loads(proof.read_text());content['archive_sha256']='0'*64;proof.write_text(json.dumps(content))
            with self.assertRaisesRegex(ValueError,'Archive hash'):verify(archive,proof,root/'bad')
            self.assertFalse((root/'bad').exists())
if __name__=='__main__':unittest.main()
