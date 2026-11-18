import json,shutil,tempfile,unittest
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from efficient_llm.warm_order.model import Config,Model,Router
from efficient_llm.warm_order.order_diagnostics import hidden,deep_features,costs,probe,PATHS,run
from efficient_llm.warm_order.token_router import run as prior_run,fit_head
from efficient_llm.warm_order.diagnostic_report import replay,report,NAMES
from efficient_llm.warm_order.data import sha
from scripts.verify_order_diagnostics_export import verify

class Diagnostics(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(9)
        self.c=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=37,prefix=3,length=12)
        self.m=Model(self.c).eval();self.x=torch.randint(0,37,(3,12))
    def test_composition_and_identity(self):
        for kind in ('attention','ffn','block'):
            torch.testing.assert_close(hidden(self.m,self.x,kind,PATHS[0]),self.m.hidden(self.x,PATHS[0]),rtol=0,atol=0)
            probe(self.m,self.x[:2],kind)
        for path in PATHS:torch.testing.assert_close(hidden(self.m,self.x,'block',path),self.m.hidden(self.x,path),rtol=0,atol=0)
        # Independent copied modules prove norms travel with the selected component.
        import copy
        for kind in ('attention','ffn'):
            for path in PATHS:
                m=copy.deepcopy(self.m)
                for j,k in enumerate(path):
                    for attr in (('self_attn','input_layernorm') if kind=='attention' else ('mlp','post_attention_layernorm')):
                        setattr(m.middle[j],attr,copy.deepcopy(getattr(self.m.middle[k],attr)))
                torch.testing.assert_close(hidden(self.m,self.x,kind,path),m.hidden(self.x,PATHS[0]),rtol=0,atol=0)
        self.assertFalse(torch.equal(hidden(self.m,self.x,'ffn',PATHS[1]),hidden(self.m,self.x,'attention',PATHS[1])))
    def test_deep_causality_and_resume(self):
        f=deep_features(self.m,self.x);x=self.x.clone();x[:,6:]=(x[:,6:]+1)%37
        torch.testing.assert_close(f[:,:4],deep_features(self.m,x)[:,:4],rtol=0,atol=0)
        self.assertFalse(f.requires_grad);probe(self.m,self.x[:2],'deep')
        c=np.random.default_rng(1).normal(size=(3,9,8)).astype('f');f=f.numpy()
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);a=fit_head(f,c,f,c,1,d/'full','cpu',epochs=2,batch=7)
            fit_head(f,c,f,c,1,d/'part','cpu',epochs=2,batch=7,stop_after=1)
            b=fit_head(f,c,f,c,1,d/'part','cpu',epochs=2,batch=7,resume=True)
            for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
    def test_real_pipeline_and_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);stem=root/'rows';rng=np.random.default_rng(4)
            rng.integers(0,37,(12,12),dtype=np.uint32).tofile(str(stem)+'.bin');starts=np.zeros((12,12),dtype='u1');starts[:,0]=1;starts.tofile(str(stem)+'.starts')
            np.savez(root/'ids.npz',fit=np.array([0,1,2]),valid=np.array([3,4]),dclm=np.array([5,6]),fwe=np.array([7,8]))
            torch.save(dict(model=self.m.state_dict(),config=asdict(self.c)),root/'weights.pt');torch.save(Router(16).state_dict(),root/'router.pt')
            hashes={str(p):sha(p) for p in root.iterdir() if p.is_file()}
            cfg=dict(name='R-s0',arm='R',seed=0,weights=str(root/'weights.pt'),router=str(root/'router.pt'),indices=str(root/'ids.npz'),stems={k:str(stem) for k in ('fit','dclm','fwe')},hashes=hashes,epochs=1,head_batch=8,batch=2)
            prior_run(cfg,root/'prior','cpu');cfg.update(arm='D',name='D-s0');prior_run(cfg,root/'dense','cpu')
            base=root/'study';(base/'runs').mkdir(parents=True)
            for name in NAMES:
                kind='deep' if '-deep-' in name else name.split('-')[0];prior=root/('prior' if kind=='deep' else 'dense')
                config=dict(cfg,name=name,kind=kind,prior=str(prior))
                run(config,base/'runs'/name,'cpu');replay(base/'runs'/name)
            (base/'inputs').mkdir();(base/'execution').mkdir()
            for p in ('inputs/complete.json','execution/source.json','execution/admission.json'):(base/p).write_text('{}')
            shutil.copyfile(root/'ids.npz',base/'inputs/indices.npz')
            result=report(base,root/'backup');self.assertEqual(result['node_raw_replay'],'PASS');self.assertEqual(len(result['comparisons']),76)
            proof=verify(base/'results.tar.gz',base/'backup.json',root/'local',target_count=9);self.assertEqual(proof['status'],'PASS')
            with self.assertRaises(ValueError):run(config,base/'runs'/name,'cpu')
            with (base/'runs/ffn-s0/dclm/costs.npy').open('ab') as f:f.write(b'bad')
            with self.assertRaises(ValueError):replay(base/'runs/ffn-s0')
if __name__=='__main__':unittest.main()
