import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from eval.a2_family_f9_v110 import select,choices_and_gold,export_items,sampler,continuous_report,render_secondary,SPEC_SHA,EVALUATOR_COMMIT
from eval.a2_family_f9 import joint_draws
from eval.downstream_suite import COUNTS
from scripts.a2_family_f9_v110_run import reuse_check
from scripts.launch_a2_family_f9_v110 import job_manifest
from run_experiments import write_json,load_jobs

class V110(unittest.TestCase):
    def test_previous_markdown_report_preserves_text_and_enforces_hash(self):
        from scripts.a2_family_f9_v110_run import locked_report
        from scripts.a2_quant_io import lock
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'REPORT.md';text='# Prior results\n\nGiữ nguyên số liệu.\n'
            p.write_text(text);identity=lock(p)
            self.assertEqual(locked_report(identity),text)
            p.write_text(text+'Changed\n')
            with self.assertRaisesRegex(ValueError,'Changed prior report'):
                locked_report(identity)

    def test_report_appends_to_locked_markdown_without_parsing_it_as_json(self):
        from scripts.a2_family_f9_v110_run import Queue
        from scripts.a2_quant_io import lock
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);prior=root/'prior.md';prior.write_text('# Prior science\n\nKept exactly.\n')
            inventory=root/'inventory.json';inventory.write_text('{"models": []}')
            q=Queue.__new__(Queue);q.root=root/'downstream';q.root.mkdir()
            q.entries=[];q.outputs={};q.flags=[]
            q.admission=dict(inventory=lock(inventory),previous_report=lock(prior),code_commit='fixture')
            with patch('eval.a2_family_f9_report.analyze',return_value={}),patch('scripts.a2_family_f9_v110_run.sampler',return_value=None),patch('scripts.a2_family_f9_v110_run.continuous_report',return_value={}),patch('eval.a2_family_f9_report.render',return_value='Family v1.9 — F9 downstream\n'),patch('eval.a2_family_f9_v110.render_secondary',return_value='Secondary metrics\n'):
                q.report()
            full=(q.root/'REPORT_FULL.md').read_text()
            self.assertTrue(full.startswith(prior.read_text()+'\n'))
            self.assertIn('Family v1.10 — F9 downstream',full)
            self.assertIn('Secondary metrics',full)
            self.assertEqual((root/'REPORT.md').read_text(),full)
            self.assertTrue((q.root/'contrasts.json').is_file())
            self.assertTrue((q.root/'continuous.json').is_file())

    def test_closed_list_actual_inventory(self):
        cfg=json.loads(Path('resources/a2_family_f9_plan_20261110.json').read_text())
        groups,excluded=select(cfg['models'])
        self.assertEqual([len(x) for x in groups],[13,12,9,12,9]);self.assertEqual(len(excluded),134)
        self.assertFalse(any(e['part'] in ('FT9','DepthB','Phase0','F2c','F8') for g in groups for e in g))
        self.assertEqual({e['trained_node'] for e in groups[1] if e['budget']==12.8},{'wc69'})
        self.assertEqual({e['budget'] for e in groups[4]},{.2,.8,3.2})

    def test_attestation_matches_original_activation_location(self):
        cfg=json.loads(Path('resources/a2_family_f9_v110_plan_20261110.json').read_text())
        expected=Path(cfg['prior_root'])/'results/a2_family_20261107/downstream/SOURCE_COMMIT_ATTESTATION.json'
        self.assertEqual(Path(cfg['evaluator_attestation']['path']),expected)
        self.assertEqual(cfg['evaluator_attestation']['sha256'],'eb3a64960edc9670e738fa9f51657107e550773e606a02fd81064d3028029d68')

    def test_winogrande_canonical_choices(self):
        c,g=choices_and_gold('winogrande',dict(doc=dict(sentence='The _ runs',option1='dog',option2='cat',answer='2')))
        self.assertEqual(c,['The dog','The cat']);self.assertEqual(g,1)

    def test_joint_continuous_bootstrap_cancellation(self):
        records={n:{t:dict(identity='same',values=(np.linspace(0,1,k)+offset).tolist()) for t,k in COUNTS.items()} for n,offset in [('a',0),('b',.2)]}
        names,d=sampler(records,128)({'b':records['b'],'a':records['a']})
        self.assertEqual(names,['b','a'])
        for p,x in d.values():np.testing.assert_allclose(x[:,0]-x[:,1],.2,atol=1e-14)
        records['b']['piqa']['identity']='wrong'
        with self.assertRaises(ValueError):sampler(records,8)

    def test_native_manifest_real_schema(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'jobs.json';write_json(p,job_manifest('/tmp/config','/tmp/root'));jobs=load_jobs(p)
            self.assertEqual(len(jobs),2)
            self.assertIn('scripts.a2_family_f9_v110_run',jobs[0]['argv'])

    def test_continuous_lambada_units_and_sign(self):
        report=dict(models={'x':{'lambada_openai':-2.,'LAMBADA_perplexity':7}},
            lead=[dict(E_down={'lambada_openai':dict(estimate=20.,ci95=[10.,30.])},accuracy_kept_share={'lambada_openai':dict(estimate=.6,ci95=[.4,.8])})],
            pooled=[],A_init_down=[{'A_init_down':{'lambada_openai':dict(estimate=10.,ci95=[5.,15.])}}],drops=[],reference=[],budget_trends=[],anneal_gains={})
        with patch('eval.a2_family_f9_report.analyze',return_value=report),patch('eval.a2_family_f9_v110.sampler',return_value=None):
            got=continuous_report([],{}, {})
        self.assertAlmostEqual(got['lead'][0]['E_down']['lambada_openai']['estimate'],.2)
        self.assertAlmostEqual(got['lead'][0]['accuracy_kept_share']['lambada_openai']['estimate'],.6)
        self.assertEqual(got['models']['x']['LAMBADA_target_NLL'],2.)

    def test_closed_queue_order_continues_failure_no_interim(self):
        from scripts.a2_family_f9_v110_run import Queue
        import threading,time
        cfg=json.loads(Path('resources/a2_family_f9_plan_20261110.json').read_text());groups,_=select(cfg['models'])
        with tempfile.TemporaryDirectory() as td:
            q=Queue.__new__(Queue);q.root=Path(td);q.groups=groups;q.entries=sum(groups,[])
            q.outputs={};q.flags=[];q.mutex=threading.Lock();q.start=time.monotonic()
            q.admission=dict(retained_flags=[],spot_groups=[[1]]*5,unusable_ids=[])
            visited=[]
            def evaluate(e,g):
                visited.append(e['model_id'])
                if e==groups[1][0]:raise ValueError('retained fixture failure')
                q.outputs[e['model_id']]={}
            q.evaluate=evaluate
            q.lanes=lambda entries,action: Queue.lanes(q,entries,action)
            with patch.object(q,'report') as report,patch.object(q,'verify_source'),patch.object(q,'verify_orchestration'):
                q.run();report.assert_called_once()
            self.assertEqual(len(visited),55);self.assertEqual(len(q.outputs),54)
            self.assertFalse((q.root/'INTERIM.md').exists())
            marker=json.loads((q.root/'complete.json').read_text())
            self.assertTrue(marker['execution_complete']);self.assertFalse(marker['coverage_complete'])
            for i in range(4):self.assertLess(max(visited.index(e['model_id']) for e in groups[i]),min(visited.index(e['model_id']) for e in groups[i+1]))

    def test_pending_native_handoff_refused_before_gpu_inspection(self):
        from scripts.launch_a2_family_f9_v110 import predecessor
        with patch('scripts.launch_a2_family_f9_v110.read',side_effect=[dict(drained=False,remaining_children={1:{}}),{},{}]),patch('scripts.launch_a2_family_f9_v110.inspect') as inspect:
            with self.assertRaises(ValueError):predecessor('/tmp/root','host')
            inspect.assert_not_called()

    def test_wait_handoff_requires_finished_receipt(self):
        from scripts.launch_a2_family_f9_v110 import wait_handoff
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'scope_v110_drain').mkdir();(root/'supervised').mkdir()
            (root/'scope_v110_drain/native_handoff_requested.json').write_text(json.dumps(dict(drained=True)))
            supervisor=root/'supervised/supervisor.json';supervisor.write_text('{}')
            def finish(_):supervisor.write_text(json.dumps(dict(finished_at='now')))
            with patch('scripts.launch_a2_family_f9_v110.time.sleep',side_effect=finish) as sleep,patch('scripts.launch_a2_family_f9_v110.predecessor',return_value={'verified':True}) as pred:
                self.assertEqual(wait_handoff(root,'host'),{'verified':True});sleep.assert_called_once();pred.assert_called_once_with(root,'host')

    def test_actual_all_tasks_choice_scores(self):
        root=Path('results/a2_depth_20261031/final_export_20261110_a01/files/dependencies')
        if not root.exists():self.skipTest('Local archived full-nine-task evidence unavailable')
        prepared=json.loads((root/'prepared.json').read_text());scored=json.loads((root/'scores/P1T-parent.json').read_text())
        replay=json.loads((root/'replays/P1T-parent.json').read_text())
        result,_=export_items(replay,prepared,scored)
        self.assertEqual({t:len(v) for t,v in result['tasks'].items()},COUNTS)
        for row in result['tasks']['lambada_openai']:
            self.assertEqual(row['target_NLL'],-row['raw_loglikelihoods'][0])
        for task in ('hellaswag','arc_easy','piqa'):
            for row in result['tasks'][task]:
                np.testing.assert_allclose(row['accuracy_scores'],np.array(row['raw_loglikelihoods'])/row['choice_character_lengths'])
        entry=scored['checkpoint_identity'];batch=scored['batch_size']
        self.assertTrue(all(reuse_check(entry,scored,prepared,batch,EVALUATOR_COMMIT).values()))
        with self.assertRaises(ValueError):reuse_check(entry,scored,prepared,batch,'wrong')
        with self.assertRaises(ValueError):reuse_check(entry,scored,prepared,batch+1,EVALUATOR_COMMIT)
        replay['samples']['piqa'][0]['filtered_resps'][0][0]+=1
        with self.assertRaises(ValueError):export_items(replay,prepared,scored)

if __name__=='__main__':unittest.main()
