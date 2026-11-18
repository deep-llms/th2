import json,tempfile,unittest
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from efficient_llm.warm_order.model import Config,Model
from efficient_llm.warm_order.data import Rows,sha
from efficient_llm.warm_order.refinement_prepare import select
from efficient_llm.warm_order.evaluate import cluster_ids
from efficient_llm.warm_order.refinement_head import fit_head
from efficient_llm.warm_order.refinement_train import train
from efficient_llm.warm_order.refinement_eval import evaluate
from efficient_llm.warm_order.refinement_report import NAMES,report,replay
from efficient_llm.warm_order.refinement_probe import numerical

class PipelineTests(unittest.TestCase):
    def test_head_resume(self):
        torch.set_num_threads(1);rng=np.random.default_rng(4);f=rng.normal(size=(4,7,16)).astype('f4');c=rng.normal(size=(4,7,2)).astype('f4')
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);a=fit_head(f,c,f[:2],c[:2],0,r/'full','cpu',epochs=3,batch=8)
            fit_head(f,c,f[:2],c[:2],0,r/'part','cpu',epochs=3,batch=8,stop_after=1)
            b=fit_head(f,c,f[:2],c[:2],0,r/'part','cpu',epochs=3,batch=8,resume=True)
            for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)
    def test_disjoint_documents(self):
        with tempfile.TemporaryDirectory() as d:
            s=str(Path(d)/'data');np.zeros((80,12),dtype='<u4').tofile(s+'.bin');starts=np.zeros((80,12),dtype='u1');starts[::2,0]=1;starts.tofile(s+'.starts');rows=Rows(s,12)
            tr,f,v=select(rows,np.arange(10),20,10,10)
            self.assertEqual(len(np.unique(np.concatenate([tr,f,v]))),40)
            self.assertEqual(len(np.intersect1d(cluster_ids(rows,f),cluster_ids(rows,v))),0)
            self.assertFalse(np.isin(np.concatenate([tr,f,v]),np.arange(10)).any())
    def test_numerical_reference(self):
        p=numerical('cpu');self.assertEqual(p['J7']['relative_gradient_error'],0)
    def test_actual_six_fits_eval_report_backup(self):
        torch.set_num_threads(1);torch.manual_seed(4)
        c=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=37,prefix=3,length=12)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);base=root/'study';(base/'inputs').mkdir(parents=True);(base/'execution').mkdir()
            stems={};hashes={}
            for domain in ('fit','dclm','fwe'):
                s=str(root/domain);stems[domain]=s;np.random.default_rng(9).integers(0,37,(24,12),dtype=np.uint32).tofile(s+'.bin');starts=np.zeros((24,12),dtype='u1');starts[:,0]=1;starts.tofile(s+'.starts')
                hashes.update({s+ext:sha(s+ext) for ext in ('.bin','.starts')})
            np.save(base/'inputs/train.npy',np.arange(8));np.savez(base/'inputs/eval.npz',fit=np.arange(8,12),valid=np.arange(12,14),dclm=np.arange(4),fwe=np.arange(4))
            ix=str(base/'inputs/eval.npz');cfg=dict(stems=stems,indices=ix,hashes={**hashes,ix:sha(ix)},final_fresh_documents=dict(dclm=True,fwe=True))
            (base/'inputs/eval-config.json').write_text(json.dumps(cfg));(base/'inputs/complete.json').write_text('{}')
            for p in ('source.json','admission.json'):(base/'execution'/p).write_text('{}')
            for name in NAMES:
                arm=name.split('-')[0];seed=int(name[-1]);torch.save(dict(config=asdict(c),model=Model(c).state_dict()),root/'initial.pt')
                tc=dict(model=asdict(c),data=stems['fit'],indices=str(base/'inputs/train.npy'),initial_weights=str(root/'initial.pt'),steps=4,batch=2,seed=seed,arm=arm,lr=.001,warmup=1,log_every=2,save_every=2)
                # Separate initial file per arm prevents later mutation of recipe inputs.
                init=root/(name+'.pt');init.write_bytes((root/'initial.pt').read_bytes());tc['initial_weights']=str(init)
                tc['expected_inputs']={p:sha(p) for p in (tc['data']+'.bin',tc['data']+'.starts',tc['indices'],tc['initial_weights'])}
                train(tc,base/'runs'/name,'cpu');evaluate(cfg,base/'runs'/name,base/'eval'/name,'cpu',head_epochs=2)
            result=report(base,root/'backup');self.assertEqual(result['node_raw_replay'],'PASS');self.assertEqual(len(result['rows']),52)
            self.assertEqual(len(json.loads((base/'backup.json').read_text())['node_copies']),17)
            from scripts.verify_adaptive_refinement_export import verify
            local=verify(base/'results.tar.gz',base/'backup.json',root/'verified');self.assertEqual(local['status'],'PASS')
            # Counterfactual corruption is detected, independent of summary metrics.
            ev=base/'eval/F7-s0';p=ev/'dclm/choices.npy';a=np.load(p);a[0,0]^=1;np.save(p,a)
            with self.assertRaises(ValueError):replay(ev)
if __name__=='__main__':unittest.main()
