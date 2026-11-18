import json,tempfile,unittest
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from efficient_llm.warm_order.model import Config,Model
from efficient_llm.warm_order.refinement_train import train
from efficient_llm.warm_order.data import sha

class TrainTests(unittest.TestCase):
    def test_actual_all_arms_resume_and_frozen_base(self):
        torch.set_num_threads(1);torch.manual_seed(4)
        c=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=37,prefix=3,length=12);m=Model(c)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);stem=root/'data';np.random.default_rng(4).integers(0,37,(8,12),dtype=np.uint32).tofile(str(stem)+'.bin');np.zeros((8,12),dtype='u1').tofile(str(stem)+'.starts');np.save(root/'ids.npy',np.arange(8));torch.save(dict(config=asdict(c),model=m.state_dict()),root/'initial.pt')
            cfg=dict(model=asdict(c),data=str(stem),indices=str(root/'ids.npy'),initial_weights=str(root/'initial.pt'),steps=4,batch=2,seed=0,lr=.001,warmup=1,log_every=1,save_every=2)
            cfg['expected_inputs']={p:sha(p) for p in (str(stem)+'.bin',str(stem)+'.starts',cfg['indices'],cfg['initial_weights'])}
            for arm in ('C6','F7','J7'):
                cfg['arm']=arm;a,ao=train(cfg,root/(arm+'full'),'cpu');train(cfg,root/(arm+'part'),'cpu',stop_after=2);b,bo=train(cfg,root/(arm+'part'),'cpu',resume=True)
                for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
                for k,v in ao.state_dict()['state'].items():
                    for key,x in v.items():torch.testing.assert_close(x,bo.state_dict()['state'][k][key],rtol=0,atol=0)
                if arm=='F7':
                    for k,v in m.state_dict().items():torch.testing.assert_close(v,a.base.state_dict()[k],rtol=0,atol=0)
                done=json.loads((root/(arm+'full')/'complete.json').read_text());self.assertEqual(done['step'],4);self.assertEqual(done['weights_sha256'],sha(root/(arm+'full')/'weights.pt'))
                with self.assertRaises(ValueError):train(cfg,root/(arm+'full'),'cpu')
                with self.assertRaises(ValueError):train(cfg,root/(arm+'part'),'cpu',resume=True)
if __name__=='__main__':unittest.main()
