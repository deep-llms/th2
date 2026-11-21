import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from efficient_llm.warm_order.model import Model, Config
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.adaptation_eval import evaluate, replay


class AdaptationEval(unittest.TestCase):
    def test_panel_replay_counts_and_tamper(self):
        torch.set_num_threads(1); torch.manual_seed(123)
        c = Config(d=8, heads=2, kv_heads=1, head_dim=4, ff=16, vocab=19, prefix=2, length=8)
        parent = Model(c).eval().requires_grad_(False)
        models = dict(parent=parent, A=copy.deepcopy(parent), B=copy.deepcopy(parent))
        # Synthetic distinct states; not labelled trained/adapted model evidence.
        with torch.no_grad():
            models['A'].middle[1].mlp.down_proj.weight.add_(.01)
            models['B'].middle[2].mlp.up_proj.weight.mul_(1.1)
        x = torch.randint(0, 19, (3, 8))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'result'
            result = evaluate(models, x, np.array([7, 4, 9]), out)
            self.assertEqual(replay(out)['cases'], 30)
            self.assertEqual(result['tokens_per_case'], 18)
            by = {r['case']: r for r in result['cases']}
            expected = float(models['B'].nll(x, (0, 2, 1, 3)).mean()-models['A'].nll(x, (0, 1, 2, 3)).mean())
            self.assertAlmostEqual(result['contrasts_nat_per_token']['remaining_gap_Bswap_minus_Acanonical'], expected, places=6)
            for branch in ('A', 'B'):
                self.assertAlmostEqual(by['weights/'+branch+'/00']['nll'], by['parent/swap']['nll'], places=10)
                self.assertAlmostEqual(by['weights/'+branch+'/11']['nll'], by[branch+'/swap']['nll'], places=10)
            self.assertEqual(set(p.name for p in out.iterdir()), {'scores.npz', 'manifest.json', 'REPORT.json'})
            report = json.loads((out/'REPORT.json').read_text()); report['cases'][0]['nll'] += .1
            (out/'REPORT.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'hash'): replay(out)
            manifest = json.loads((out/'manifest.json').read_text())
            manifest['files']['REPORT.json'] = sha(out/'REPORT.json')
            (out/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'replay mismatch'): replay(out)
            with self.assertRaisesRegex(ValueError, 'Fresh'): evaluate(models, x, np.array([7, 4, 9]), out)
            with self.assertRaisesRegex(ValueError, 'Unique'): evaluate(models, x, np.array([7, 7, 9]), Path(tmp)/'dup')


if __name__ == '__main__': unittest.main()
