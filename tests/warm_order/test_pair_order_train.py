import json
import torch
import numpy as np
from pathlib import Path
from tests.warm_order.test_frozen_router_train import FrozenRouterTrainTests
from efficient_llm.warm_order.pair_order_train import train
from efficient_llm.warm_order.pair_order import load_initial
from efficient_llm.warm_order.data import sha


class PairOrderTrainTests(FrozenRouterTrainTests):
    def setUp(self):
        super().setUp()
        self.cfg.update(schedule='paired-body-router-warmup-cosine', router_lr=.01)

    def test_actual_entrypoint_all_arms_exact_resume_and_immutable_body(self):
        # Override the fixture's frozen-router test: this study MUST update the body.
        for arm in ('C','G','M'):
            c={**self.cfg,'arm':arm,'name':arm+'-s0'}
            initial=load_initial(c)
            whole=self.root/(arm+'whole'); part=self.root/(arm+'part')
            m,o=train(c,whole,'cpu')
            # Interrupt before warmup completes so both schedule phases are crossed.
            train(c,part,'cpu',stop_after=1)
            n,oo=train(c,part,'cpu',resume=True)
            self.assert_same(m.state_dict(),n.state_dict()); self.assert_same(o.state_dict(),oo.state_dict())
            self.assertFalse(torch.equal(initial.embed.weight,m.embed.weight))
            self.assertEqual(len(o.param_groups),1 if arm=='C' else 2)
            for g in o.param_groups:
                self.assertEqual(g['weight_decay'],.1 if g['group_name']=='body' else 0.)
            if arm!='C': self.assertFalse(torch.equal(initial.pair_logits,m.pair_logits))
            a=torch.load(whole/'fullstate.pt',weights_only=False)
            b=torch.load(part/'fullstate.pt',weights_only=False)
            for key in ('torch_rng','python_rng','counts','block_tokens','step'):self.assert_same(a[key],b[key])
            np.testing.assert_array_equal(a['numpy_rng'][1],b['numpy_rng'][1])
            done=json.loads((part/'complete.json').read_text())
            self.assertEqual(done['block_tokens'],6*2*12*10)
            self.assertEqual(done['step'],6)
            for name,digest in c['parent_hashes'].items():self.assertEqual(sha(Path(c['parent'])/name),digest)

    def test_resume_rejects_changed_recipe_data_and_checkpoint(self):
        out=self.root/'part';train(self.cfg,out,'cpu',stop_after=3)
        with self.assertRaises(ValueError):train({**self.cfg,'router_lr':.02},out,'cpu',resume=True)
        p=out/'fullstate.pt';p.write_bytes(p.read_bytes()+b'bad')
        with self.assertRaises(ValueError):train(self.cfg,out,'cpu',resume=True)
        ids=Path(self.cfg['indices']);np.save(ids,np.arange(12,dtype=np.int64)[::-1])
        with self.assertRaises(ValueError):train(self.cfg,self.root/'changed','cpu')


if __name__=='__main__':
    import unittest
    unittest.main()
