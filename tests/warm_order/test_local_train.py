import copy,json,os,subprocess,sys,tempfile,unittest
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from efficient_llm.warm_order.data import Rows,sha
from efficient_llm.warm_order.model import Config,Model
from efficient_llm.warm_order.local_policy import LocalRouter
from efficient_llm.warm_order.local_train import train,update
from efficient_llm.warm_order.local_prepare import fresh_test_ids


def fixture(p,arm):
    torch.manual_seed(0);c=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=32,prefix=3,length=12)
    torch.save(dict(config=asdict(c),model=Model(c).state_dict()),p/'initial.pt')
    rng=np.random.default_rng(9)
    rng.integers(32,size=(64,12),dtype='u4').tofile(p/'data.bin')
    b=np.zeros((64,12),dtype='u1');b[:,0]=1;b.tofile(p/'data.starts')
    np.save(p/'ids.npy',np.arange(24));np.savez(p/'eval.npz',fit=np.arange(24,32),dclm=np.arange(32,48),fwe=np.arange(48,64))
    return dict(data=str(p/'data'),indices=str(p/'ids.npy'),initial_weights=str(p/'initial.pt'),
                expected_inputs={str(p/f):sha(p/f) for f in ('data.bin','data.starts','ids.npy','initial.pt')},
                model=asdict(c),arm=arm,seed=0,batch=4,steps=6,lr=.001,router_lr=.002,exploration=.1,
                warmup=2,threads=1,save_every=3,log_every=1)


def equal_tree(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,atol=0,rtol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal_tree(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal_tree(x,y)
    else:assert a==b,(a,b)


class LocalTrainTests(unittest.TestCase):
    def test_all_arms_fullstate_resume_and_recipe_guards(self):
        for arm in ('C','U','CF','PG'):
            with self.subTest(arm=arm),tempfile.TemporaryDirectory() as d:
                p=Path(d);c=fixture(p,arm)
                train(c,p/'whole','cpu');train(c,p/'split','cpu',stop_after=2)
                with self.assertRaisesRegex(ValueError,'recipe'):
                    train(dict(c,exploration=.2),p/'split','cpu',resume=True)
                train(c,p/'split','cpu',resume=True)
                a=torch.load(p/'whole/fullstate.pt',weights_only=False);b=torch.load(p/'split/fullstate.pt',weights_only=False)
                for key in ('model','router','optimizer','head_optimizer','step','counts','compute','sample_rng','torch_rng'):
                    equal_tree(a[key],b[key])
                self.assertEqual(sum(a['counts']),24)
                self.assertEqual(a['compute']['ce_block_tokens'],24*12*6)
                self.assertEqual(a['compute']['extra_forward_block_tokens'],24*12*6*{'C':0,'U':0,'PG':1,'CF':4}[arm])
                with self.assertRaisesRegex(ValueError,'Already complete'):train(c,p/'split','cpu',resume=True)
                with self.assertRaisesRegex(ValueError,'Nonempty'):train(c,p/'whole','cpu')

    def test_labels_and_body_share_preupdate_weights(self):
        torch.set_num_threads(1)
        c=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=32,prefix=3,length=12)
        for arm in ('PG','CF'):
            model=Model(c);router=LocalRouter(16)
            old={k:p.clone() for k,p in model.state_dict().items()};original=model.nll;calls=[]
            def checked(x,path):
                for k,p in model.state_dict().items():torch.testing.assert_close(p,old[k],atol=0,rtol=0)
                calls.append(tuple(path));return original(x,path)
            model.nll=checked
            update(model,router,torch.optim.AdamW(model.parameters(),lr=.001),torch.optim.AdamW(router.parameters(),lr=.001),
                   torch.randint(32,(4,12)),arm,.1,torch.Generator().manual_seed(3))
            self.assertGreaterEqual(len(calls),2 if arm=='PG' else 5)
            self.assertTrue(any(not torch.equal(p,old[k]) for k,p in model.state_dict().items()))

    def test_fresh_test_excludes_entire_pilot_documents(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'rows';np.zeros((20,12),dtype='u4').tofile(str(p)+'.bin')
            starts=np.zeros((20,12),dtype='u1');starts[::4,0]=1;starts.tofile(str(p)+'.starts')
            rows=Rows(p,12);chosen=fresh_test_ids(rows,np.array([1,9]),maximum=20,minimum=1)
            self.assertEqual(set(chosen),set(range(4,8))|set(range(12,20)))
            with self.assertRaises(ValueError):fresh_test_ids(rows,np.array([1,5,9,13,17]),minimum=1)

    def test_real_cli_train_eval(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);c=fixture(p,'PG');(p/'config.json').write_text(json.dumps(c))
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1'}
            def run(module,*args):subprocess.run([sys.executable,'-m',module,*args],env=env,check=True,capture_output=True,text=True)
            run('efficient_llm.warm_order.local_train','--config',str(p/'config.json'),'--out',str(p/'run'),'--device','cpu')
            ev=dict(run=str(p/'run'),fit=str(p/'data'),dclm=str(p/'data'),fwe=str(p/'data'),indices=str(p/'eval.npz'),
                    input_hashes={str(p/f):sha(p/f) for f in ('data.bin','data.starts','eval.npz')})
            (p/'eval.json').write_text(json.dumps(ev))
            run('efficient_llm.warm_order.local_evaluate','--config',str(p/'eval.json'),'--out',str(p/'evaluation'),'--device','cpu')
            report=json.loads((p/'evaluation/report.json').read_text())
            for domain in ('dclm','fwe'):
                a=np.load(p/f'evaluation/{domain}_costs.npz');expected=a['costs'][np.arange(16),a['choices']]
                self.assertAlmostEqual(float(expected.mean()),report['domains'][domain]['metrics']['learned'])
                self.assertEqual(sum(report['domains'][domain]['choices']),16)


if __name__=='__main__':unittest.main()
