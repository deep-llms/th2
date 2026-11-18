import json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from efficient_llm.warm_order.train import train,lr_at

class Resume(unittest.TestCase):
    def test_fullstate_exact_and_guards(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);rng=np.random.default_rng(8)
            rng.integers(0,32,(48,16),dtype=np.uint32).tofile(p/'train.bin');np.zeros((48,16),dtype=np.uint8).tofile(p/'train.starts')
            np.save(p/'ids.npy',rng.permutation(48))
            c=dict(data=str(p/'train'),indices=str(p/'ids.npy'),batch=2,steps=24,arm='W',seed=4,lr=.001,warmup=3,threads=1,save_every=12,log_every=1,model=dict(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=32,prefix=4,length=16))
            m,o=train(c,p/'whole','cpu');train(c,p/'split','cpu',stop_after=5)
            bad=dict(c,lr=.002)
            with self.assertRaisesRegex(ValueError,'recipe'):train(bad,p/'split','cpu',resume=True)
            m2,o2=train(c,p/'split','cpu',resume=True)
            for k,v in m.state_dict().items():torch.testing.assert_close(v,m2.state_dict()[k],rtol=0,atol=0)
            for k,v in o.state_dict()['state'].items():
                for key,t in v.items():torch.testing.assert_close(t,o2.state_dict()['state'][k][key],rtol=0,atol=0)
            t1=[json.loads(x) for x in (p/'whole/trace.jsonl').read_text().splitlines()];t2=[json.loads(x) for x in (p/'split/trace.jsonl').read_text().splitlines()]
            for a,b in zip(t1,t2):
                for key in ('step','loss','lr','path','first_row','block_tokens'):self.assertEqual(a[key],b[key])
            self.assertEqual(len(t1),len(t2))
            for key in ('input_tokens','scored_tokens','block_tokens','counts'):
                self.assertEqual(json.loads((p/'whole/complete.json').read_text())[key],json.loads((p/'split/complete.json').read_text())[key])
            with self.assertRaisesRegex(ValueError,'Already complete'):train(c,p/'split','cpu',resume=True)
            with self.assertRaisesRegex(ValueError,'Nonempty'):train(c,p/'whole','cpu')
    def test_lr_endpoints(self):
        self.assertAlmostEqual(lr_at(0,100,.01,10),.001)
        self.assertAlmostEqual(lr_at(9,100,.01,10),.01)
        self.assertAlmostEqual(lr_at(99,100,.01,10),.001)
if __name__=='__main__':unittest.main()
