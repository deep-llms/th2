"""Branch/ordering tests: reuse parent scores, prohibit scoring before decisions."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from eval.a2_run import write,lock
from eval.a2_request_bridge import digest
from eval.a2_stage1b_final_run import run_screen
from eval.a2_stage1b_stats import PART_MODELS
from eval.downstream_suite import COUNTS
from scripts.launch_a2_stage1b_final_downstream import job_config,budget
from run_experiments import load_jobs


class FinalRun(unittest.TestCase):
    def exercise(self,root,batch=2,fail_gate=False,fail_report=False,fail_pairs=False,unlimited=False):
        identities={name:dict(name=name) for name in PART_MODELS['final']}
        prepared=dict(requests=[],protocol={})
        plan=dict(identities={k:write(root/(k+'.json'),v) for k,v in identities.items()},
            prepared=write(root/'prepared.json',prepared),native_python='/native/python',eval_python='/eval/python',
            tokenizer='/tok',dataset_root='/data',manifest={'path':'/manifest'},input_spec={'path':'/input'},
            tasks=list(COUNTS),gpus=list(range(8)),policy={},time_policy={},failed_gate={})
        events=[]
        def command(argv,**kw):
            module=argv[2];events.append(module)
            target=Path(argv[argv.index('--output')+1]);target.parent.mkdir(parents=True,exist_ok=True)
            self.assertIsNone(kw['timeout']) if unlimited else self.assertGreater(kw['timeout'],0)
            if module=='eval.a2_stage1b_check':
                self.assertEqual(kw['env']['CUDA_VISIBLE_DEVICES'],'0')
                identity=json.loads(Path(argv[argv.index('--identity')+1]).read_text())
                value=dict(passed=not fail_gate,production_batch_size=2,
                    checkpoint_identity_sha256=digest(identity),prepared_sha256=digest(prepared))
            elif module=='eval.a2_stage1b_batch_report':
                self.assertEqual(kw['env']['CUDA_VISIBLE_DEVICES'],'')
                if fail_report:raise ValueError('invalid comparison')
                value=dict(status='compared',requests=78259,hard_batch_passed=True,production_batch_size=batch,batch2_accepted=batch==2,
                    inputs=json.loads(Path(argv[argv.index('--inputs')+1]).read_text()))
            else:
                self.assertEqual(module,'eval.a2_requests');self.assertEqual(argv[3],'replay')
                self.assertEqual(kw['env']['CUDA_VISIBLE_DEVICES'],'');value={}
            write(target,value)
        def scoring(jobs,**kw):
            is_parent=jobs[0]['argv'][2]=='eval.a2_stage1b_parent_score'
            events.append('parent' if is_parent else 'production')
            self.assertEqual(events.count('eval.a2_stage1b_check'),6)
            if is_parent:
                self.assertEqual(len(jobs),2);self.assertNotIn('eval.a2_stage1b_batch_report',events)
                self.assertEqual({j['gpu'] for j in jobs},{0,1})
            else:
                self.assertEqual(len(jobs),5);self.assertIn('eval.a2_stage1b_batch_report',events)
                self.assertNotIn('P300-parent',[j['name'] for j in jobs])
            for job in jobs:
                argv=job['argv'];actual=int(argv[argv.index('--batch-size')+1])
                if not is_parent:self.assertEqual(actual,batch)
                write(job['output'],dict(batch_size=actual))
                if is_parent:write(argv[argv.index('--predictions-output')+1],{})
        with patch('eval.a2_stage1b_final_run.validate_plan',return_value=identities), \
             patch('eval.a2_stage1b_final_run.subprocess.run',side_effect=command), \
             patch('eval.a2_stage1b_final_run.execute_final_scoring',side_effect=scoring), \
             patch('eval.a2_stage1b_final_run.collect_files',return_value=dict(status='ok',diagnostic=False)), \
             patch('eval.a2_stage1b_final_run.paired_report',return_value=dict(resamples=10000,seed=0),
                   side_effect=ValueError('unpaired') if fail_pairs else None):
            if fail_gate or fail_report or fail_pairs:
                with self.assertRaises(ValueError):run_screen(plan,root/'out',timeout_seconds=None if unlimited else 60)
                self.assertFalse((root/'out/complete.json').exists())
                if fail_gate:self.assertNotIn('parent',events)
                if fail_report:self.assertNotIn('production',events)
                return
            run_screen(plan,root/'out',timeout_seconds=None if unlimited else 60)
        done=json.loads((root/'out/complete.json').read_text())
        self.assertEqual(done['models'],6);self.assertEqual(done['production_batch_size'],batch)
        self.assertEqual(done['scores']['P300-parent'],lock(root/'out/batch_comparison'/str(batch)/'scores.json'))
        self.assertEqual(events.count('eval.a2_requests'),7) # two parent replays + other five
        self.assertEqual(len(done['revised_gates']),6)

    def test_batch2_reuses_parent_and_batch1_fallback_all_models(self):
        for batch in (1,2):
            with self.subTest(batch=batch),tempfile.TemporaryDirectory() as d:self.exercise(Path(d),batch)

    def test_authorized_unlimited_runtime_has_no_subprocess_timeout(self):
        with tempfile.TemporaryDirectory() as d:self.exercise(Path(d),unlimited=True)

    def test_failures_never_publish_complete_or_skip_decision(self):
        for case in ('fail_gate','fail_report','fail_pairs'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as d:self.exercise(Path(d),**{case:True})

    def test_unlimited_scoring_uses_only_supplied_lanes_and_no_timeout(self):
        from eval.a2_stage1b_final_run import execute_final_scoring
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'scores.json';job=dict(gpu=3,output=str(target),argv=['scorer'])
            def child(argv,**kwargs):
                self.assertEqual(argv,['scorer']);self.assertNotIn('timeout',kwargs)
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'3')
                write(target,{})
            with patch('eval.a2_stage1b_final_run.subprocess.run',side_effect=child):
                execute_final_scoring([job],timeout_seconds=None)
            self.assertTrue(target.is_file())
            with self.assertRaises(ValueError):execute_final_scoring([job],timeout_seconds=None)

    def test_native_contract_is_six_models_and_full_comparison(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'jobs.json';write(p,job_config(dict(path='/plan',sha256='a'*64)))
            job=load_jobs(p)[0]
            self.assertIsNone(job['timeout_seconds']);self.assertEqual(job['gpus'],list(range(8)))
            required={x['path']:x['json_equals'] for x in job['required_outputs']}
            self.assertEqual(required['downstream/complete.json']['models'],6)
            self.assertEqual(required['downstream/batch_decision.json']['requests'],78259)


if __name__=='__main__':unittest.main()
