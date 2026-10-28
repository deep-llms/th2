import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from efficient_llm.a2.protocol import Run,FIXED_SETS,DATASETS,shared_check_sets,layers_for
from scripts.a2_remaining import remaining_runs
from scripts.a2_remaining_heals import execute_remaining,verify_result
from tests.a2.test_remaining import completed_s1,row_for,provenance,survey_fixture


class RemainingHealsTests(unittest.TestCase):
    def fixture(self,root):
        _,_,assets=survey_fixture();sets={d:dict(FIXED_SETS,S2=[2,3,4]) for d in DATASETS}
        def put(name,value):
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
            return hashlib.sha256(p.read_bytes()).hexdigest()
        prov=provenance();sh=put('sets.json',{'sets':sets})
        for name,key,value in [('assets.json','assets_sha256',assets),('secondary.json','secondary_map_sha256',{}),
            ('data/manifest.json','data_sha256',{})]:prov[key]=put(name,value)
        patterns=[layers_for(g) for g in shared_check_sets(sets,assets).values()]+[[0],[11],[23]]
        check=dict(passed=True,scope='shared_mechanism',representative='dclm/P300',
            sets_sha256=sh,provenance=prov,removal_patterns=patterns)
        gate=dict(status='passed',launch_scope='all',provenance=prov,sets_sha256=sh,shared_numerical=check,
            tests={t:{'passed':True} for t in ('T4','T5','T6','T7','T8','T10_formula','T11','T12','T13')},
            parents={d+'/'+p:{'passed':True} for d in DATASETS for p in ('P300','P1T')})
        put('gate.json',gate)
        pre={k:dict(provenance=prov,sets_sha256=sh,parent_sanity={'passed':True},
            parent={'final':{'loss':2.}},sets={s:{'final':{'loss':3.}} for s in sets[k.split('/')[0]]}) for k in gate['parents']}
        common=['--preflight',str(root/'gate.json'),'--sets',str(root/'sets.json'),'--assets',str(root/'assets.json'),
            '--data',str(root/'data'),'--secondary-map',str(root/'secondary.json')]
        return dict(root=root/'run',launcher=['python','native'],common=common,history={'rows':completed_s1()},
            pre=pre,gate=gate,sets=sets,assets=assets,sets_sha256=sh,estimated={b:100 for b in ('0.05B','0.2B','0.8B')},
            spent_seconds=16000)

    def fake_child(self,prov,calls,fail_first=False,infra=False):
        def run(argv,**kw):
            r=json.loads(Path(argv[argv.index('--run')+1]).read_text());calls.append(r)
            native=Run(**{k:r[k] for k in ('dataset','arm','set','budget','lr','suffix')})
            row=dict(row_for(native),provenance=prov,rerun_of=r['rerun_of'])
            if infra: row.update(status='failed',diverged=False)
            elif fail_first and len(calls)==1:row.update(status='failed',diverged=True)
            out=Path(argv[argv.index('--output')+1]);out.mkdir();(out/'result.json').write_text(json.dumps(row))
            return SimpleNamespace(returncode=1 if row['status']=='failed' else 0)
        return run

    def test_only_28_missing_fits_no_history_relabel_or_repeat(self):
        with tempfile.TemporaryDirectory() as d:
            args=self.fixture(Path(d));calls=[];history=copy.deepcopy(args['history'])
            with patch('scripts.a2_remaining_heals.subprocess.run',side_effect=self.fake_child(args['gate']['provenance'],calls)):
                result=execute_remaining(**args)
            self.assertTrue(result['full_phase_complete']);self.assertEqual(len(calls),28)
            self.assertEqual(args['history'],history)
            self.assertFalse(set(result['reused_ids']) & {r['run_id'] for r in calls})
            self.assertEqual({r['run_id'] for r in calls},{r['run_id'] for r in remaining_runs(history['rows'])['pending']})

    def test_one_half_lr_retry_preserves_t14_suffix(self):
        with tempfile.TemporaryDirectory() as d:
            args=self.fixture(Path(d));calls=[]
            r=next(r for r in remaining_runs(args['history']['rows'])['pending'] if r['suffix'])
            with patch('scripts.a2_remaining_heals.subprocess.run',side_effect=self.fake_child(args['gate']['provenance'],calls,True)):
                result=execute_remaining(**args,subset=[r['run_id']])
            self.assertEqual(len(calls),2);self.assertEqual(calls[1]['lr'],r['lr']/2)
            self.assertEqual(calls[1]['suffix'],r['suffix']+'-rerun')
            self.assertEqual(calls[1]['rerun_of'],r['run_id']);self.assertEqual(result['status'],'complete')
            self.assertFalse(result['full_phase_complete'])

    def test_completed_but_tuning_diverged_retries_then_stays_incomplete(self):
        # Native heal returns status=complete even when finite tuning loss is
        # worse than unhealed; this is registered divergence, not infrastructure.
        with tempfile.TemporaryDirectory() as d:
            args=self.fixture(Path(d));calls=[]
            chosen=remaining_runs(args['history']['rows'])['pending'][:2]
            def child(argv,**kw):
                response=self.fake_child(args['gate']['provenance'],calls)(argv,**kw)
                out=Path(argv[argv.index('--output')+1])/'result.json'
                row=json.loads(out.read_text())
                if len(calls)<=2:row.update(diverged=True,val_tune_pre=2.,val_tune_post=3.)
                out.write_text(json.dumps(row));return response
            with patch('scripts.a2_remaining_heals.subprocess.run',side_effect=child):
                result=execute_remaining(**args,subset=[r['run_id'] for r in chosen])
            self.assertEqual(len(calls),3)
            self.assertEqual(calls[1]['lr'],chosen[0]['lr']/2)
            self.assertEqual(result['status'],'incomplete')
            self.assertFalse(result['scientific_complete'])
            self.assertEqual(result['completed_original_ids'],[chosen[1]['run_id']])

    def test_infra_no_retry_and_caps_or_reused_subset_refused(self):
        with tempfile.TemporaryDirectory() as d:
            args=self.fixture(Path(d));calls=[];pending=remaining_runs(args['history']['rows'])['pending']
            with self.assertRaises(ValueError):execute_remaining(**args,subset=[args['history']['rows'][0]['run_id']])
            with self.assertRaises(ValueError):execute_remaining(**dict(args,spent_seconds=64700))
            with patch('scripts.a2_remaining_heals.subprocess.run',side_effect=self.fake_child(args['gate']['provenance'],calls,infra=True)):
                with self.assertRaises(ValueError):execute_remaining(**args,subset=[pending[0]['run_id']])
            self.assertEqual(len(calls),1)

    def test_result_schedule_and_coverage_fail_closed(self):
        r=Run('dclm','P300','S2','0.05B',3e-4);row=row_for(r)
        verify_result(row,r,provenance(),None,0)
        for k,v in [('lr_updates',[0]*r.steps),('coverage',{}),('roundtrip_difference',.1),('delta_post',1.)]:
            with self.subTest(k=k),self.assertRaises(ValueError):verify_result(dict(row,**{k:v}),r,provenance(),None,0)
