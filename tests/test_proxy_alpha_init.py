"""Gate-only initialization and the real HF checkpoint/resume path."""
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import load_file
from tests.test_proxy_heads import model, batch
from tests.test_proxy_training import proxy_fixture
from tests.test_train import invoke


class AlphaInitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_only_gates_change_and_lm_trains_proxy_immediately(self):
        for arm in ('P1-block', 'P3-block'):
            zero=model(arm);one=model(arm,alpha_init=1.)
            for name,value in zero.state_dict().items():
                expected=torch.ones_like(value) if name.endswith('.alpha') else value
                torch.testing.assert_close(one.state_dict()[name],expected,rtol=0,atol=0)
            out=one(batch(),compute_auxiliary_losses=False)
            loss=out['lm_sum']/out['lm_count'];self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for head in one.heads.values():
                for p in head.parameters():
                    self.assertTrue(torch.isfinite(p.grad).all())
                    self.assertGreater(p.grad.abs().sum(),0)
            self.assertEqual(one.auxiliary_weight(250),zero.auxiliary_weight(250))
        for value in (float('nan'),float('inf')):
            with self.assertRaisesRegex(ValueError,'alpha_init'):model('P1-block',alpha_init=value)
        with self.assertRaisesRegex(ValueError,'alpha_init'):model('A',alpha_init=1.)

    def test_real_resume_and_default_recipe_compatibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);recipe=proxy_fixture(root)
            args={**recipe,'arm':'P1-block','proxy_alpha_init':1.,'output_dir':str(root/'one')}
            invoke(root,{**args,'stop_after':2})
            saved=json.loads((root/'one/train_config.json').read_text())
            self.assertEqual(saved['pilot']['proxy_alpha_init'],1.)
            with self.assertRaises(ValueError):invoke(root,{**args,'proxy_alpha_init':0.})
            invoke(root,args)
            invoke(root,{**args,'output_dir':str(root/'full')})
            resumed=load_file(root/'one/model.safetensors');full=load_file(root/'full/model.safetensors')
            for name in full:torch.testing.assert_close(resumed[name],full[name],rtol=0,atol=0)
            default={**recipe,'arm':'P1-block','output_dir':str(root/'zero')}
            invoke(root,{**default,'stop_after':2})
            self.assertNotIn('proxy_alpha_init',json.loads((root/'zero/train_config.json').read_text())['pilot'])
            invoke(root,default)  # Old recipes omit the option; resume must still work.
