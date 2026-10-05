import json
from pathlib import Path
import tempfile
import unittest

import torch
from scripts.evaluate_proxy_gates import run_sweep, scaled_gates, state_hash
from tests.test_proxy_heads import model, batch
from tests.test_proxy_training import proxy_fixture
from tests.test_train import invoke


class GateSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_scales_only_gates_and_restores_after_exception(self):
        for arm in ('P1-block','P3-block'):
            m=model(arm).eval()
            with torch.no_grad():
                for head in m.heads.values():head.alpha.uniform_(-.1,.1)
                before={k:v.clone() for k,v in m.state_dict().items()}
                with self.assertRaisesRegex(RuntimeError,'interrupt'):
                    with scaled_gates(m,5):
                        for k,v in m.state_dict().items():
                            torch.testing.assert_close(v,before[k]*5 if k.endswith('.alpha') else before[k],rtol=0,atol=0)
                        raise RuntimeError('interrupt')
                for k,v in m.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
                with scaled_gates(m,0):zero=m(batch())['lm_sum']
                with m.without_proxy():disabled=m(batch())['lm_sum']
                torch.testing.assert_close(zero,disabled,rtol=0,atol=0)
            with self.assertRaisesRegex(ValueError,'evaluation-only'):
                with scaled_gates(m,2):pass

    def test_real_trainer_saved_checkpoint_evaluation(self):
        for arm in ('P1-block','P3-block'):
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);recipe=proxy_fixture(root)
                source=root/arm
                invoke(root,{**recipe,'arm':arm,'output_dir':str(source),'stop_after':2})
                run_sweep(source,root/'evaluation',step=2)
                receipt=json.loads((root/'evaluation/summary.json').read_text())
                self.assertEqual(receipt['status'],'passed')
                self.assertEqual([r['multiplier'] for r in receipt['measurements']],[1,2,5,1])
                self.assertTrue(receipt['model_state_unchanged'])
                self.assertFalse((root/'evaluation/optimizer.pt').exists())
