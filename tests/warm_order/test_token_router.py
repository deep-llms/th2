import json,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from efficient_llm.warm_order.model import Config,Model,Router,ORDERS
from efficient_llm.warm_order.token_router import stem_features,token_cost,expected_loss,fit_head,id_selector,summarize,probe
from efficient_llm.warm_order.token_prepare import reserve
from efficient_llm.warm_order.data import Rows,sha
from efficient_llm.warm_order.token_report import replay_run,report,NAMES

class TokenRouter(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(9)
        self.c=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=37,prefix=3,length=12)
        self.m=Model(self.c).eval();self.x=torch.randint(0,37,(3,12))
    def test_target_shift_and_sequence_parity(self):
        for path in ORDERS:
            loss=token_cost(self.m,self.x,path)
            logits=self.m.hidden(self.x,path)@self.m.embed.weight.T
            reference=torch.nn.functional.cross_entropy(logits[:,2:-1].reshape(-1,37),self.x[:,3:].reshape(-1),reduction='none').reshape(3,9)
            torch.testing.assert_close(loss,reference)
            torch.testing.assert_close(loss.mean(1),self.m.nll(self.x,path))
        probe(self.m,self.x[:2])
    def test_no_future_or_target_in_features(self):
        r=Router(16);h=stem_features(self.m,self.x);changed=self.x.clone();changed[:,6:]=(changed[:,6:]+1)%37
        # h index3 is position5, whose target is position6, changed here.
        hc=stem_features(self.m,changed)
        torch.testing.assert_close(h[:,:4],hc[:,:4],rtol=0,atol=0)
        torch.testing.assert_close(r(h[:,:4]),r(hc[:,:4]),rtol=0,atol=0)
        self.assertFalse(torch.equal(h[:,4:],hc[:,4:]))
        r(h).sum().backward();self.assertTrue(all(p.grad is None for p in self.m.parameters()))
    def test_loss_gradient_and_detached_cost(self):
        z=torch.zeros(4,8,requires_grad=True);c=torch.ones(4,8,requires_grad=True);c.data[:,3]=0
        expected_loss(z,c).backward();self.assertIsNone(c.grad)
        self.assertTrue((z.grad[:,3]<0).all());self.assertTrue((z.grad[:,0]>0).all())
    def test_id_selector_uses_current_token_and_fallback(self):
        t=np.array([[1,2,1]]);c=np.ones((1,3,8));c[0,[0,2],3]=0;c[0,1,5]=0
        a=id_selector(t,c,10,smoothing=1);self.assertEqual(a[1],3);self.assertEqual(a[2],5);self.assertEqual(a[9],3)
    def test_reductions(self):
        c=np.random.default_rng(4).uniform(0,5,(2,3,8)).astype(np.float32);ch=np.array([[0,1,2],[5,6,7]])
        v=summarize(c,{'learned':ch},2)
        expected=np.array([sum(c[i,j,ch[i,j]] for j in range(3))/3 for i in range(2)])
        np.testing.assert_allclose(v['learned'],expected)
        np.testing.assert_allclose(v['ensemble'],-np.log(np.exp(-c.astype(float)).mean(-1)).mean(1))
        self.assertTrue((v['oracle']<=v['learned']).all())
    def test_fullstate_resume_identical(self):
        rng=np.random.default_rng(5);x=rng.normal(size=(3,9,16)).astype('f');c=rng.normal(size=(3,9,8)).astype('f')
        with tempfile.TemporaryDirectory() as d:
            full=fit_head(x,c,x,c,13,Path(d)/'full','cpu',epochs=3,batch=7)
            fit_head(x,c,x,c,13,Path(d)/'part','cpu',epochs=3,batch=7,stop_after=1)
            resumed=fit_head(x,c,x,c,13,Path(d)/'part','cpu',epochs=3,batch=7,resume=True)
            for a,b in zip(full.parameters(),resumed.parameters()):torch.testing.assert_close(a,b,atol=0,rtol=0)
            f=torch.load(Path(d)/'full/fullstate.pt',weights_only=False);r=torch.load(Path(d)/'part/fullstate.pt',weights_only=False)
            with self.assertRaises(ValueError):fit_head(x+1,c,x,c,13,Path(d)/'part','cpu',epochs=3,batch=7,resume=True)
            self.assertEqual(f['trace'],r['trace']);self.assertTrue(torch.equal(f['generator'],r['generator']))
            for k,v in f['optimizer']['state'].items():
                for key,value in v.items():torch.testing.assert_close(value,r['optimizer']['state'][k][key],rtol=0,atol=0)
            with self.assertRaises(ValueError):fit_head(x,c,x,c,13,Path(d)/'part','cpu',epochs=4,batch=7,resume=True)
    def test_real_cpu_entrypoint(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);stem=root/'rows';rng=np.random.default_rng(4)
            a=rng.integers(0,37,(24,12),dtype=np.uint32);b=np.zeros((24,12),dtype='u1');b[::2,0]=1
            a.tofile(str(stem)+'.bin');b.tofile(str(stem)+'.starts')
            fit,valid=reserve(Rows(stem,12),np.array([0,1]),fit_n=3,valid_n=2)
            self.assertEqual(len(set(fit)&set(valid)),0)
            np.savez(root/'ids.npz',fit=fit,valid=valid,dclm=np.array([20,21]),fwe=np.array([22,23]))
            torch.save(dict(model=self.m.state_dict(),config=asdict(self.c)),root/'weights.pt')
            torch.save(Router(16).state_dict(),root/'router.pt')
            hashes={str(p):sha(p) for p in root.iterdir() if p.is_file()}
            cfg=dict(name='R-s0',arm='R',seed=0,weights=str(root/'weights.pt'),router=str(root/'router.pt'),indices=str(root/'ids.npz'),stems={k:str(stem) for k in ['fit','dclm','fwe']},hashes=hashes,epochs=2,head_batch=8,batch=2)
            (root/'config.json').write_text(json.dumps(cfg))
            argv=[sys.executable,'-m','efficient_llm.warm_order.token_router','--config',str(root/'config.json'),'--out',str(root/'run'),'--device','cpu']
            p=subprocess.run(argv,capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr)
            replay_run(root/'run')
            proof=json.loads((root/'run/complete.json').read_text());self.assertEqual(proof['status'],'complete')
            for file,want in proof['files'].items():self.assertEqual(sha(root/'run'/file),want)
            for domain in ['dclm','fwe']:
                c=np.load(root/'run'/domain/'costs.npy');cs={k:np.load(root/'run'/domain/(k+'.npy')) for k in ['learned','prefix','token_id']}
                got=np.load(root/'run'/domain/'summary.npz');const=json.loads((root/'run/selection.json').read_text())['constant']
                for k,v in summarize(c,cs,const).items():np.testing.assert_array_equal(got[k],v)
            p=subprocess.run(argv,capture_output=True,text=True);self.assertNotEqual(p.returncode,0)
            cfg.update(name='D-s0',arm='D');(root/'configD.json').write_text(json.dumps(cfg))
            argvD=[sys.executable,'-m','efficient_llm.warm_order.token_router','--config',str(root/'configD.json'),'--out',str(root/'dense'),'--device','cpu']
            p=subprocess.run(argvD,capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr)
            base=root/'experiment';(base/'runs').mkdir(parents=True)
            for name in NAMES:shutil.copytree(root/('dense' if name.startswith('D') else 'run'),base/'runs'/name)
            (base/'inputs').mkdir();(base/'execution').mkdir()
            for rel in ['inputs/complete.json','execution/source.json','execution/admission.json']:(base/rel).write_text('{}')
            shutil.copyfile(root/'ids.npz',base/'inputs/indices.npz')
            result=report(base,root/'backup');self.assertEqual(result['node_raw_replay'],'PASS')
            self.assertEqual(len(result['comparisons']),32)
            proof=json.loads((base/'backup.json').read_text());self.assertEqual(sha(base/'results.tar.gz'),proof['archive_sha256'])
            for file,want in proof['files'].items():self.assertEqual(sha(base/file),want)
            # Replay must reject any mutation, not just trust the marker.
            with (base/'runs/R-s0/dclm/costs.npy').open('ab') as f:f.write(b'bad')
            with self.assertRaises(ValueError):replay_run(base/'runs/R-s0')

if __name__=='__main__':unittest.main()
