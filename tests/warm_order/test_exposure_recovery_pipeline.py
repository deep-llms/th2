import json,tempfile,unittest
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from tests.warm_order import test_exposure_recovery as fixtures
from efficient_llm.warm_order.model import Config
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.exposure_recovery_prepare import prepare,NAMES
from efficient_llm.warm_order.exposure_recovery_eval import run
from efficient_llm.warm_order.exposure_recovery_report import report,replay_run
from scripts.verify_exposure_recovery_export import verify


class PipelineTests(unittest.TestCase):
    def test_ten_fits_disjoint_rows_parent_identity_replay_21backups_tamper(self):
        self.exercise(False)

    def test_extension_ten_fit_pipeline(self):
        self.exercise(True)

    def exercise(self, extension):
        torch.set_num_threads(1)
        model=Config(d=8,heads=1,kv_heads=1,head_dim=8,ff=16,vocab=19,length=8,prefix=2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);hashes={};stems={};parents={}
            for d in ('fit','dclm','fwe'):
                stem=str(root/d);stems[d]=stem
                np.random.default_rng(3).integers(0,19,(80,8),dtype=np.uint32).tofile(stem+'.bin')
                starts=np.zeros((80,8),dtype='u1');starts[:,0]=1;starts.tofile(stem+'.starts')
                hashes.update({stem+s:sha(stem+s) for s in ('.bin','.starts')})
            for name in NAMES:
                p=root/'parents'/name;p.mkdir(parents=True)
                _,_,pins,_=fixtures.RecoveryTests().fixture(p,name[0],name.split('-')[0].endswith('F'),int(name[-1]),stems['fit'])
                parents[name]=dict(directory=str(p),hashes=pins)
                hashes.update({str(p/k):v for k,v in pins.items()})
            np.savez(root/'eval.npz',fit=np.arange(12,16),valid=np.arange(16,20),dclm=np.arange(4),fwe=np.arange(4))
            np.save(root/'old.npy',np.arange(12))
            for p in (root/'eval.npz',root/'old.npy'):hashes[str(p)]=sha(p)
            plan=dict(hashes=hashes,parents=parents,parent_steps=3,
                reserved=[dict(path=str(root/'eval.npz'),keys=['fit','valid'])],
                excluded_orders={str(s):[str(root/'old.npy')] for s in (0,1)},
                training=dict(data=stems['fit'],model=asdict(model),batch=2,steps=4,log_every=1,save_every=2,threads=1),
                eval=dict(stems=stems,indices=str(root/'eval.npz')))
            if extension:
                manifest=dict(tokenizer_sha256='fixture',partition='fixture-heal-vs-final')
                (root/'manifest.json').write_text(json.dumps(manifest))
                prefix=dict(tokenizer_sha256='fixture',partition=manifest['partition'],
                    source_manifest_sha256=sha(root/'manifest.json'),streams={'train':{'tokens':20*8}})
                (root/'prefix.json').write_text(json.dumps(prefix))
                for p in (root/'manifest.json',root/'prefix.json'):hashes[str(p)]=sha(p)
                plan['extension']=dict(source_stem=stems['fit'],manifest=str(root/'manifest.json'),
                    prefix_manifest=str(root/'prefix.json'),prefix_expected=prefix,prefix_stem=stems['fit'],
                    first_token=20*8,stop_token=80*8)
            base=root/'study';prepare(plan,base/'inputs');(base/'execution').mkdir()
            for p in ('source.json','admission.json'):(base/'execution'/p).write_text('{}')
            before={p:sha(p) for p in hashes}
            prepared=json.loads((base/'inputs/complete.json').read_text())
            self.assertTrue(all(v['prior_row_overlap']==v['reserved_doc_overlap']==0 for v in prepared['reservation'].values()))
            if extension:
                selected=np.load(base/'inputs/source_rows.npy')
                self.assertTrue((selected>=20).all());self.assertEqual(len(np.unique(selected)),16)
                x=np.fromfile(base/'inputs/recovery_train.bin',dtype='<u4').reshape(-1,8)
                original=np.fromfile(stems['fit']+'.bin',dtype='<u4').reshape(-1,8)
                np.testing.assert_array_equal(x,original[selected])
            for name in NAMES:run(json.loads((base/f'inputs/{name}.json').read_text()),base/'runs'/name,'cpu')
            result=report(base,root/'backup')
            self.assertEqual(before,{p:sha(p) for p in hashes})
            self.assertEqual(len(result['rows']),40)
            self.assertEqual(len(json.loads((base/'backup.json').read_text())['node_copies']),21)
            self.assertEqual(verify(base/'results.tar.gz',base/'backup.json',root/'verified')['status'],'PASS')
            with self.assertRaises(ValueError):prepare(plan,base/'inputs')
            a=dict(np.load(base/'runs/SF-s0/dclm/summary.npz'));a['soft_counts'][0,0,0]+=1
            np.savez(base/'runs/SF-s0/dclm/summary.npz',**a)
            with self.assertRaises(ValueError):replay_run(base/'runs/SF-s0')
            with (base/'results.tar.gz').open('ab') as f:f.write(b'tamper')
            with self.assertRaises(ValueError):verify(base/'results.tar.gz',base/'backup.json',root/'bad-export')

if __name__=='__main__':unittest.main()
