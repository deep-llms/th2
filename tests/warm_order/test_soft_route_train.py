import json,tempfile,unittest
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from efficient_llm.warm_order.model import Config,Model
from efficient_llm.warm_order.soft_route_train import train
from efficient_llm.warm_order.data import sha


class ResumeTests(unittest.TestCase):
    def test_all_arms_fullstate_and_accounting(self):
        torch.set_num_threads(1)
        c=Config(d=8,heads=1,kv_heads=1,head_dim=8,ff=16,vocab=19,length=8,prefix=2)
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);stem=str(r/'data')
            np.random.default_rng(4).integers(0,19,(12,8),dtype=np.uint32).tofile(stem+'.bin')
            starts=np.zeros((12,8),dtype='u1');starts[:,0]=1;starts.tofile(stem+'.starts')
            np.save(r/'ids.npy',np.arange(12))
            torch.manual_seed(9)
            torch.save(dict(config=asdict(c),arm='D',seed=0,model=Model(c).state_dict()),r/'parent.pt')
            for arm,blocks in (('C',6),('G',18),('S',18)):
                cfg=dict(model=asdict(c),arm=arm,seed=0,batch=2,steps=6,lr=.0003,router_lr=.001,warmup=2,
                    data=stem,indices=str(r/'ids.npy'),initial_weights=str(r/'parent.pt'),canonical_probability=.999,
                    log_every=1,save_every=3,threads=1)
                cfg['expected_inputs']={p:sha(p) for p in (stem+'.bin',stem+'.starts',cfg['indices'],cfg['initial_weights'])}
                a,ao=train(cfg,r/(arm+'whole'),'cpu')
                train(cfg,r/(arm+'resume'),'cpu',stop_after=3)
                b,bo=train(cfg,r/(arm+'resume'),'cpu',resume=True)
                for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
                self.assertEqual(ao.state_dict()['param_groups'],bo.state_dict()['param_groups'])
                for k,state in ao.state_dict()['state'].items():
                    for key,value in state.items():torch.testing.assert_close(value,bo.state_dict()['state'][k][key],rtol=0,atol=0)
                one=torch.load(r/(arm+'whole')/'fullstate.pt',weights_only=False)
                two=torch.load(r/(arm+'resume')/'fullstate.pt',weights_only=False)
                torch.testing.assert_close(one['torch_rng'],two['torch_rng'],rtol=0,atol=0)
                self.assertEqual(one['counts'],two['counts']);self.assertEqual(one['block_tokens'],6*2*8*blocks)
                for mutation in (dict(router_lr=.002),dict(canonical_probability=.99)):
                    with self.assertRaises(ValueError):train(dict(cfg,**mutation),r/(arm+'resume'),'cpu',resume=True)

if __name__=='__main__':unittest.main()
