import copy
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from efficient_llm.warm_order.model import Config
from efficient_llm.warm_order.soft_route import Model as Parent
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.soft_exploration_train import write_json
from efficient_llm.warm_order.frozen_router_train import train


class FrozenRouterTrainTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(19)
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.c=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=37,length=12,prefix=3)
        parent=self.root/'parent';parent.mkdir();m=Parent(self.c,'dense')
        pc=dict(name='C-s0',arm='C',seed=0,model=asdict(self.c),steps=2000)
        recipe=dict(config=pc)
        torch.save(dict(config=asdict(self.c),arm='C',seed=0,canonical_probability=.999,model=m.state_dict()),parent/'weights.pt')
        torch.save(dict(recipe=recipe,step=2000,model=m.state_dict()),parent/'fullstate.pt')
        write_json(parent/'recipe.json',recipe)
        write_json(parent/'complete.json',dict(name='C-s0',arm='C',seed=0,step=2000,
            weights_sha256=sha(parent/'weights.pt'),fullstate_sha256=sha(parent/'fullstate.pt')))
        data=str(self.root/'data');rng=np.random.default_rng(2)
        rng.integers(0,37,(12,12),dtype=np.uint32).tofile(data+'.bin')
        starts=np.zeros((12,12),dtype=np.uint8);starts[:,0]=1;starts.tofile(data+'.starts')
        ids=self.root/'rows.npy';np.save(ids,np.arange(12,dtype=np.int64))
        self.cfg=dict(name='M-s0',arm='M',seed=0,model=asdict(self.c),canonical_probability=.99,router_width=8,
            parent=str(parent),parent_steps=2000,parent_hashes={f:sha(parent/f) for f in ('weights.pt','fullstate.pt','complete.json','recipe.json')},
            data=data,indices=str(ids),steps=6,batch=2,lr=.003,warmup=2,threads=1,log_every=1,save_every=3,
            schedule='router-only-warmup-cosine',expected_inputs={p:sha(p) for p in (data+'.bin',data+'.starts',str(ids))})

    def tearDown(self):self.tmp.cleanup()

    def assert_same(self,a,b):
        if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
        elif isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for k in a:self.assert_same(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.assert_same(x,y)
        else:self.assertEqual(a,b)

    def test_actual_entrypoint_all_arms_exact_resume_and_immutable_body(self):
        for arm in ('G','L','M'):
            c={**self.cfg,'arm':arm,'name':arm+'-s0'}
            whole=self.root/(arm+'whole');part=self.root/(arm+'part')
            m,o=train(c,whole,'cpu');train(c,part,'cpu',stop_after=3);n,oo=train(c,part,'cpu',resume=True)
            self.assert_same(m.state_dict(),n.state_dict());self.assert_same(o.state_dict(),oo.state_dict())
            done=json.loads((part/'complete.json').read_text());self.assertTrue(done['frozen_body_exact'])
            self.assertEqual(done['step'],6)
            self.assertEqual(json.loads((part/'resume.json').read_text())['resumed_step'],3)
            for p,h in c['parent_hashes'].items():self.assertEqual(sha(Path(c['parent'])/p),h)
            self.assertEqual(len(o.state_dict()['state']),len(m.router_parameters()))

    def test_resume_rejects_changed_recipe_data_and_checkpoint(self):
        out=self.root/'part';train(self.cfg,out,'cpu',stop_after=3)
        with self.assertRaises(ValueError):train({**self.cfg,'lr':.01},out,'cpu',resume=True)
        p=out/'fullstate.pt';p.write_bytes(p.read_bytes()+b'bad')
        with self.assertRaises(ValueError):train(self.cfg,out,'cpu',resume=True)
        ids=Path(self.cfg['indices']);np.save(ids,np.arange(12,dtype=np.int64)[::-1])
        with self.assertRaises(ValueError):train(self.cfg,self.root/'changed','cpu')


if __name__=='__main__':unittest.main()
