import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from efficient_llm.warm_order.data import sha
class Entry(unittest.TestCase):
    def test_train_eval_cli(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);rng=np.random.default_rng(19)
            for split in ('train','dev','test'):
                rng.integers(0,32,(64,16),dtype=np.uint32).tofile(p/(split+'.bin'))
                starts=np.zeros((64,16),dtype='u1');starts[:,0]=1;starts.tofile(p/(split+'.starts'))
            np.save(p/'ids.npy',np.arange(16));np.savez(p/'eval.npz',fit=np.arange(16,48),dev=np.arange(16),test=np.arange(16))
            cfg=dict(data=str(p/'train'),indices=str(p/'ids.npy'),batch=2,steps=8,arm='W',seed=0,lr=.001,warmup=2,threads=1,save_every=4,log_every=1,model=dict(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=32,prefix=4,length=16))
            (p/'config.json').write_text(json.dumps(cfg));env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1'}
            def run(module,*args):
                subprocess.run([sys.executable,'-m',module,*args],env=env,check=True,capture_output=True,text=True)
            run('efficient_llm.warm_order.train','--config',str(p/'config.json'),'--out',str(p/'run'),'--device','cpu')
            cfg=dict(run=str(p/'run'),fit=str(p/'train'),dev=str(p/'dev'),test=str(p/'test'),indices=str(p/'eval.npz'))
            cfg['input_hashes']={str(f):sha(f) for f in p.iterdir() if f.suffix in ('.bin','.starts','.npz')}
            (p/'eval.json').write_text(json.dumps(cfg))
            # Production row length is512; tiny fixture evaluator uses checkpoint length.
            run('efficient_llm.warm_order.evaluate','--config',str(p/'eval.json'),'--out',str(p/'evaluation'),'--device','cpu')
            result=json.loads((p/'evaluation/report.json').read_text());self.assertEqual(result['scored_tokens'],16*12)
            arrays=np.load(p/'evaluation/test_costs.npz');self.assertAlmostEqual(result['metrics']['learned'],float(arrays['learned'].mean()))
            for k,v in json.loads((p/'evaluation/complete.json').read_text())['files'].items():self.assertEqual(sha(p/'evaluation'/k),v)
if __name__=='__main__':unittest.main()
