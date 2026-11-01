import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from tests.a2.test_model import tiny_model
from efficient_llm.a2.model import bypass
from efficient_llm.a2.measure import score
from efficient_llm.a2_depth.common import layer_sets, removed_layers, write, Checks, DATASETS, PARENTS, SETS
from scripts.a2_depth_parta_report import compute


class DepthTests(unittest.TestCase):
    def test_windows_and_phase0_groups(self):
        groups={ds:dict(S2=[1,2,5] if ds=='nem' else [1,2,3],S3=[3,6,9],S4=[1,2,6]) for ds in DATASETS}
        sets=layer_sets(groups)
        self.assertEqual(sets['dclm']['W8'],list(range(16,22)))
        self.assertEqual(sets['nem']['S2'],[2,3,4,5,10,11])
        for ds in DATASETS:
            self.assertEqual(set(sets[ds]),set(SETS))
            self.assertTrue(all(len(v)==6 and min(v)>=2 and max(v)<24 for v in sets[ds].values()))

    def test_native_physical_removal_matches_hook_and_restores(self):
        model=tiny_model(num_hidden_layers=24)
        rows=[dict(input_ids=torch.tensor([1,2,3,4,5]),sample_id=0)]
        original=score(model,rows,amp=False,distributed=False)
        for layers in (list(range(2,8)),list(range(16,22)),list(range(18,24)),[2,3,4,5,10,11]):
            with bypass(model,layers):expected=score(model,rows,amp=False,distributed=False)
            with removed_layers(model,layers) as kept:
                actual=score(model,rows,amp=False,distributed=False)
                self.assertEqual(kept,[i for i in range(24) if i not in layers])
                self.assertEqual([l.self_attn.layer_idx for l in model.model.layers],kept)
            np.testing.assert_allclose(actual['per_sequence_nll'],expected['per_sequence_nll'],rtol=0,atol=1e-7)
            self.assertEqual(score(model,rows,amp=False,distributed=False),original)
            self.assertFalse(hasattr(model.config,'a2_kept_layers'))
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with removed_layers(model,list(range(2,8))):raise RuntimeError('fixture')
        self.assertEqual(score(model,rows,amp=False,distributed=False),original)

    def test_failure_is_recorded_and_next_check_runs(self):
        with tempfile.TemporaryDirectory() as d:
            checks=Checks(Path(d)/'tests.json')
            self.assertIsNone(checks.attempt('bad',lambda:1/0))
            self.assertEqual(checks.attempt('next',lambda:42),42)
            self.assertFalse(json.loads(checks.path.read_text())['checks'][0]['passed'])

    def test_paired_readout_and_missing_scope(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg=dict(resamples=100,deviations=[])
            base=np.array([3.,4.,5.,6.]);spread=np.array([.7,.9,1.1,1.3])
            for ds in DATASETS:
                for j,p in enumerate(PARENTS):
                    for k,s in enumerate(('parent',)+SETS):
                        # All paired damage vectors are proportional: relative CI is exact.
                        damage=0 if s=='parent' else (.1+.01*j+.002*k)*spread
                        arr=base-.02*j+damage
                        write(root/'part_a'/ds/p/(s+'.json'),dict(per_sequence_nll=arr.tolist(),loss=float(arr.mean())))
            result,checks,runs=compute(cfg,root)
            self.assertTrue(result['coverage_complete']);self.assertEqual(len(runs),156)
            self.assertTrue(all(c['passed'] for c in checks))
            r=result['datasets']['dclm']['W1']['contrasts']['R_short']
            self.assertAlmostEqual(r['estimate'],.01)
            self.assertAlmostEqual(r['relative_to_reference_damage'],.01/.102)
            np.testing.assert_allclose(r['relative_ci95'],[.01/.102]*2)
            self.assertFalse(result['predictions_nats']['predictions']['A3'])
            (root/'part_a'/'nem'/'P1T'/'W9.json').unlink()
            result,_,_=compute(cfg,root)
            self.assertFalse(result['coverage_complete'])
            self.assertIsNone(result['predictions_nats']['predictions']['A1'])
            self.assertIn('W1',result['datasets']['nem'])

    def test_coordinator_failure_continues_all_twelve(self):
        import scripts.a2_depth_parta_run as runner
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg=root/'plan.json';write(cfg,dict(results_root=d));calls=[]
            def fake(cmd,**kwargs):
                calls.append(cmd)
                return SimpleNamespace(returncode=1 if cmd[-1]=='P300' else 0)
            with patch('sys.argv',['runner','--config',str(cfg),'--receipt',str(root/'done.json')]),patch.object(runner.subprocess,'run',side_effect=fake),patch('scripts.a2_depth_parta_report.finalize',return_value=dict(coverage_complete=False)):
                runner.main()
            self.assertEqual(len(calls),12)
            done=json.loads((root/'done.json').read_text())
            self.assertTrue(done['execution_complete']);self.assertFalse(done['coverage_complete'])

if __name__=='__main__':unittest.main()
