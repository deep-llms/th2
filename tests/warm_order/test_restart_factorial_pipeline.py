import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from efficient_llm.warm_order.model import Config
from efficient_llm.warm_order.soft_route import Model as SoftModel
from efficient_llm.warm_order.soft_route_eval import Reference as SoftReference
from efficient_llm.warm_order.soft_exploration_eval import collect
from efficient_llm.warm_order.data import Rows,sha
from efficient_llm.warm_order.train import write_json
from efficient_llm.warm_order.restart_factorial_prepare import prepare
from efficient_llm.warm_order.restart_factorial_eval import run
from efficient_llm.warm_order.pair_order_eval import run as old_run
from efficient_llm.warm_order.exposure_recovery import optimizer_for
from efficient_llm.warm_order.restart_factorial_report import NAMES,report,replay_run
from scripts.verify_restart_factorial_export import verify


class RestartFactorialPipelineTests(unittest.TestCase):
    def test_six_fit_actual_pipeline_controls_hashes_replay_backup_and_tamper(self):
        torch.set_num_threads(1)
        model=Config(d=8,heads=1,kv_heads=1,head_dim=8,ff=16,vocab=19,length=8,prefix=2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);hashes={};stems={};parents={};controls={};orders={}
            for domain in ('fit','dclm','fwe'):
                stem=str(root/domain);stems[domain]=stem
                np.random.default_rng(1).integers(0,19,(40,8),dtype=np.uint32).tofile(stem+'.bin')
                starts=np.zeros((40,8),dtype=np.uint8);starts[:,0]=1;starts.tofile(stem+'.starts')
                hashes.update({stem+s:sha(stem+s) for s in ('.bin','.starts')})
            np.savez(root/'eval.npz',fit=np.arange(20,24),valid=np.arange(24,28),dclm=np.arange(4),fwe=np.arange(4))
            hashes[str(root/'eval.npz')]=sha(root/'eval.npz')
            for seed in (0,1):
                torch.manual_seed(seed);name=f'C-s{seed}';p=root/'parent'/name/'train';p.mkdir(parents=True)
                m=SoftModel(model,'dense');recipe=dict(config=dict(name=name,arm='C',seed=seed,model=asdict(model),steps=3,parent_steps=2,schedule='inherit-terminal-lr-constant-no-floor'),parent_recipe=dict(config=dict(arm='C',steps=2,lr=1e-4,canonical_probability=.999)))
                opt=optimizer_for(m,1e-4,1e-3)
                for _ in range(5):
                    opt.zero_grad();m.nll(torch.randint(19,(2,8))).mean().backward();opt.step()
                opt.param_groups[0]['lr']=1e-5
                torch.save(dict(config=asdict(model),arm='C',seed=seed,canonical_probability=.999,model=m.state_dict()),p/'weights.pt')
                torch.save(dict(recipe=recipe,step=3,model=m.state_dict(),optimizer=opt.state_dict()),p/'fullstate.pt')
                write_json(p/'recipe.json',recipe)
                done=dict(name=name,arm='C',seed=seed,step=3,weights_sha256=sha(p/'weights.pt'),fullstate_sha256=sha(p/'fullstate.pt'))
                write_json(p/'complete.json',done)
                pins={f:sha(p/f) for f in ('weights.pt','fullstate.pt','complete.json','recipe.json')}
                parents[str(seed)]=dict(directory=str(p),hashes=pins);hashes.update({str(p/f):h for f,h in pins.items()})
                ref=SoftReference(model,'dense');ref.load_state_dict(m.state_dict());m.eval();ref.eval();arrays={};members={}
                for d in ('dclm','fwe'):
                    target=p.parent/d/'summary.npz';target.parent.mkdir()
                    a,_=collect(m,ref,Rows(stems[d],8),np.arange(4),'cpu');np.savez_compressed(target,**a)
                    arrays[d]=str(target);hashes[str(target)]=sha(target);members[d+'/summary.npz']=sha(target)
                complete=p.parent/'complete.json';write_json(complete,dict(status='complete',seed=seed,train=done,files=members));hashes[str(complete)]=sha(complete)
                controls[name.replace('C-','P-')]=dict(complete=str(complete),arrays=arrays)
                order=root/f'rows{seed}.npy';np.save(order,np.arange(seed*8,(seed+1)*8));orders[str(seed)]=str(order);hashes[str(order)]=sha(order)
            plan=dict(hashes=hashes,parents=parents,parent_steps=3,controls=controls,orders=orders,
                training=dict(data=stems['fit'],model=asdict(model),steps=4,batch=2,lr=.003,router_lr=.01,warmup=1,
                    canonical_probability=.99,router_width=8,threads=1,log_every=1,save_every=2,schedule='dense-restart-factorial',terminal_lr=1e-5),
                eval=dict(stems=stems,indices=str(root/'eval.npz')))
            # Historical FH is produced by the unchanged real Pair C entrypoint.
            for seed in (0,1):
                parent=parents[str(seed)]
                tc=dict(plan['training'],name=f'C-s{seed}',arm='C',seed=seed,parent=parent['directory'],parent_hashes=parent['hashes'],parent_steps=3,indices=orders[str(seed)])
                tc.pop('terminal_lr');tc['schedule']='paired-body-router-warmup-cosine'
                tc['expected_inputs']={p:sha(p) for p in (tc['data']+'.bin',tc['data']+'.starts',tc['indices'])}
                dest=root/f'old/FH-s{seed}'
                old_run(dict(name=tc['name'],train=tc,eval=plan['eval'],hashes=dict(hashes)),dest,'cpu')
                files={d:str(dest/d/'summary.npz') for d in ('dclm','fwe')}
                done=json.loads((dest/'complete.json').read_text())
                controls[f'FH-s{seed}']=dict(complete=str(dest/'complete.json'),recipe=str(dest/'train/recipe.json'),arrays=files,fullstate_sha256=done['train']['fullstate_sha256'])
                for p in [*files.values(),str(dest/'complete.json'),str(dest/'train/recipe.json')]:hashes[p]=sha(p)
            base=root/'study';prepare(plan,base/'inputs');(base/'execution').mkdir()
            for f in ('source.json','admission.json'):write_json(base/'execution'/f,{})
            before={p:sha(p) for p in hashes}
            for name in NAMES:run(json.loads((base/f'inputs/{name}.json').read_text()),base/'runs'/name,'cpu')
            r=report(base,root/'backup');self.assertEqual(len(r['rows']),40)
            self.assertEqual(len(json.loads((base/'backup.json').read_text())['node_copies']),13)
            self.assertEqual(verify(base/'results.tar.gz',base/'backup.json',root/'replay')['status'],'PASS')
            self.assertEqual(before,{p:sha(p) for p in hashes})
            with self.assertRaises(ValueError):prepare(plan,base/'inputs')
            damaged=base/'runs/IL-s0/dclm/summary.npz';a=dict(np.load(damaged));a['soft'][0]+=1;np.savez(damaged,**a)
            with self.assertRaises(ValueError):replay_run(base/'runs/IL-s0')
            with (base/'results.tar.gz').open('ab') as f:f.write(b'bad')
            with self.assertRaises(ValueError):verify(base/'results.tar.gz',base/'backup.json',root/'bad')


if __name__=='__main__':unittest.main()
