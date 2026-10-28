import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from efficient_llm.a2.pair import SCOPE,reuse_record,verify_grid,validate_result
from efficient_llm.a2.protocol import Run,LRS,validate_run_scope,shared_check_sets,FIXED_SETS
from scripts.a2_preflight_inputs import initial_asset_plan,pair_jobs
from run_experiments import load_jobs

ROOT=Path(__file__).resolve().parents[2]

class PairTests(unittest.TestCase):
    def test_scope_and_chosen_lr(self):
        gate=dict(launch_scope=SCOPE,chosen_lrs={'P300':3e-4,'P1T':1e-4})
        valid=[Run('dclm','P1T','S1','0.03B',lr) for lr in LRS]
        valid += [Run('dclm',arm,'S1',budget,gate['chosen_lrs'][arm])
                  for arm in ('P300','P1T') for budget in ('0.05B','0.2B')]
        for r in valid:
            validate_run_scope(r,gate,SCOPE)
            retry=Run(r.dataset,r.arm,r.set,r.budget,r.lr/2,'-rerun')
            validate_run_scope(retry,gate,SCOPE,r.run_id)
            with self.assertRaises(ValueError):validate_run_scope(retry,gate,SCOPE)
        for r in (Run('dclm','P300','S1','0.03B',3e-4),Run('fwe','P1T','S1','0.2B',1e-4),
                  Run('dclm','P1T','S2','0.2B',1e-4),Run('dclm','P1T','S1','0.2B',3e-4)):
            with self.assertRaises(ValueError):validate_run_scope(r,gate,SCOPE)

    def test_receipt_reuse_preserves_original_and_rejects_changes(self):
        original=json.loads((ROOT/'docs/results/a2_early_grid_20261028/check.json').read_text())
        prov=copy.deepcopy(original['provenance']);prov['code_commit']='a'*40
        prov['code_files']['a2/pair.py']='new';prov['code_files']['a2/protocol.py']='scoping-only'
        result=reuse_record(original,prov,'new-set-scope')
        self.assertEqual(result['historical_receipt'],original)
        self.assertNotEqual(result['provenance'],original['provenance'])
        for key in ('data_sha256','software','python'):
            bad=copy.deepcopy(prov);bad[key]='changed'
            with self.assertRaises(ValueError):reuse_record(original,bad,'new')
        bad=copy.deepcopy(prov);bad['code_files']['a2/train.py']='changed'
        with self.assertRaises(ValueError):reuse_record(original,bad,'new')

    def test_pinned_grid_and_pair_asset_coverage(self):
        plan=json.loads((ROOT/'resources/a2_inputs_20261028.json').read_text())
        initial=initial_asset_plan(plan);pair=initial_asset_plan(plan,('P300','P1T'))
        paths={d['path'] for d in pair['downloads']}
        self.assertTrue({d['path'] for d in initial['downloads']}<=paths)
        for parent in ('P300','P1T'):
            asset=plan['assets']['dclm/'+parent]
            for name in asset['files']:self.assertIn(str(Path(asset['weights_dir'])/name),paths)
        shared_check_sets({'dclm':{'S1':FIXED_SETS['S1']}},plan['assets'],scope=SCOPE)
        # Tracked archive contains every exact reused receipt needed by verifier.
        lock=json.loads((ROOT/'resources/a2_pair_reuse.json').read_text())
        from efficient_llm.a2.early_grid import ast_hash
        import hashlib
        for name,digest in lock['unchanged_files'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest)
        for name,defs in lock['unchanged_definitions'].items():
            for fn,digest in defs.items():self.assertEqual(ast_hash(ROOT/name,fn),digest)

    def test_failure_contract(self):
        r=Run('dclm','P1T','S1','0.03B',1e-4)
        result=dict(r.record(),provenance={'x':1},rerun_of=None,status='complete',
                    coverage={'exact_order':True},roundtrip_difference=0)
        validate_result(result,r,{'x':1},None,0)
        with self.assertRaises(ValueError):validate_result(result,r,{'x':1},None,1)
        with self.assertRaises(ValueError):validate_result(dict(result,coverage={'exact_order':False}),r,{'x':1},None,0)
        failed=dict(result,status='failed',diverged=True)
        validate_result(failed,r,{'x':1},None,1)
        with self.assertRaises(ValueError):validate_result(dict(failed,diverged=False),r,{'x':1},None,1)

    def test_bounded_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'jobs.json';p.write_text(json.dumps({'jobs':pair_jobs('/inputs','/prior','/grid',12345)}))
            jobs=load_jobs(p)
            self.assertIn('efficient_llm.a2.pair',json.loads(p.read_text())['jobs'][0]['argv'])

if __name__=='__main__':unittest.main()
