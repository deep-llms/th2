import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from efficient_llm.a2_depth.common import Checks,SPEC_SHA,read,write
from scripts.a2_depth_score import identity
from scripts.a2_depth_downstream import paired,verify_replay,run


class DepthDownstreamTests(unittest.TestCase):
    def test_saved_checkpoint_identity_and_incomplete_schedule_flag(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'heals/ST1-dclm-P300-S1-12.8B-lr1e-3'
            write(folder/'result.json',dict(dataset='dclm',arm='P300',block='S1',lr_updates=[0.]*48828,
                                          unverified=False,flags=[]))
            write(folder/'recipe.json',dict(steps=48828,removed=list(range(16,22))))
            (folder/'final').mkdir();(folder/'final/model.safetensors').write_bytes(b'fixture')
            cfg=dict(results_root=d,spec_sha256=SPEC_SHA,source_manifest={})
            checks=Checks(root/'checks.json');a=identity(cfg,'P300',checks)
            self.assertTrue(all(e['passed'] for e in checks.items))
            (folder/'final/model.safetensors').write_bytes(b'changed')
            b=identity(cfg,'P300',checks);self.assertNotEqual(a['files'],b['files'])
            result=read(folder/'result.json');result['lr_updates']=[];write(folder/'result.json',result)
            identity(cfg,'P300',checks)
            self.assertFalse(checks.items[-2]['passed'])

    def test_complete_pairing_and_fresh_item_tamper(self):
        from tests.a2.test_stage1_stats import Stage1ReplayPairingTest
        from eval.downstream_suite import PRIMARY
        a=Stage1ReplayPairingTest().fixture();b=copy.deepcopy(a)
        self.assertTrue(verify_replay(a,b));self.assertEqual(paired(a),paired(b))
        # Same aggregate and bootstrap distribution, but different per-item correctness.
        task='hellaswag';name='P300-student';key=PRIMARY[task].split(',')[0]
        x,y=b[name]['samples'][task][:2];x[key],y[key]=y[key],x[key]
        self.assertFalse(verify_replay(a,b))

    def test_failed_scores_do_not_cancel_other_arms_or_fabricate_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=dict(results_root=d,prior_downstream={'scores':{},'replays':{}},downstream_common={},prepared={})
            calls=[]
            def fake(args,**kwargs):calls.append(args);return SimpleNamespace(returncode=1)
            with patch('scripts.a2_depth_partc_run.require_free_gpus') as guard,patch('scripts.a2_depth_downstream.subprocess.run',side_effect=fake):
                result=run(cfg,'config')
            self.assertEqual(guard.call_count,3)
            self.assertEqual([a[a.index('--arm')+1] for a in calls],['P300','P1T','X72'])
            self.assertTrue(result['execution_complete']);self.assertFalse(result['coverage_complete'])
            self.assertIsNone(result['paired']);self.assertTrue(result['flags'])


if __name__=='__main__':unittest.main()
