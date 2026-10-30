import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from eval.a2_run import write,lock
from eval.a2_request_bridge import digest
from eval.a2_stage1b_stats import PART_MODELS
from scripts.a2_stage1b_option2 import run
from scripts.launch_a2_stage1b_final_downstream import job_config
from run_experiments import load_jobs


class Option2(unittest.TestCase):
    def exercise(self,root,fail_score=False,fail_pair=False):
        names=PART_MODELS['final'];ids={n:dict(name=n) for n in names}
        prepared=dict(requests=[dict(context=[1],continuation=[2])]*78259)
        pl=write(root/'prepared.json',prepared)
        idlocks={n:write(root/(n+'.json'),v) for n,v in ids.items()}
        parent=write(root/'old-parent-score.json',dict(batch_size=2,checkpoint_identity_sha256=digest(ids['P300-parent']),prepared_sha256=digest(prepared)))
        parent_replay=write(root/'old-parent-replay.json',{})
        gates={n:write(root/(n+'-gate.json'),{}) for n in names}
        decision=write(root/'decision.json',dict(hard_batch_passed=False,full_max_nats_per_token=.10355663299560547,
            full_greedy_token_mismatch_rate=.0009372587936927997,batch2_accepted=True,
            inputs=dict(scores={'2':parent},replays={'2':parent_replay})))
        reuse=write(root/'reuse.json',dict(policy=write(root/'policy.json',{}),decision=decision,gates=gates))
        plan=dict(identities=idlocks,prepared=pl,continuation=reuse,native_python='/native/python',eval_python='/eval/python',
            tokenizer='/tok',input_spec={'path':'/input'},dataset_root='/data',manifest={'path':'/manifest'},tasks=['task'],gpus=list(range(8)),time_policy={})
        events=[]
        def command(argv,**kw):
            module=argv[2];events.append((module,argv))
            self.assertFalse(kw['check']);self.assertNotIn('timeout',kw)
            target=Path(argv[argv.index('--output')+1])
            if module=='eval.a2_score':
                self.assertEqual(argv[argv.index('--batch-size')+1],'2')
                identity=json.loads(Path(argv[argv.index('--identity')+1]).read_text())
                self.assertNotEqual(identity['name'],'P300-parent')
                if fail_score and identity['name']=='P1T-parent':return type('R',(),dict(returncode=1))()
                write(target,dict(batch_size=2,checkpoint_identity_sha256=digest(identity),prepared_sha256=digest(prepared),scores=[[-1,False]]*78259))
            elif module=='eval.a2_requests':write(target,{})
            elif module=='scripts.verify_a2_stage1b_option2':write(target,dict(passed=not fail_score and not fail_pair))
            else:self.fail(module)
            return type('R',(),dict(returncode=0))()
        with patch('scripts.a2_stage1b_option2.validate_plan',return_value=ids), \
             patch('scripts.verify_a2_stage1b_final.verify_gate',side_effect=ValueError('reported gate issue')), \
             patch('scripts.a2_stage1b_option2.collect_files',return_value=dict(status='ok',diagnostic=False,scores={})), \
             patch('scripts.a2_stage1b_option2.paired_report',return_value=dict(resamples=10000,seed=0),side_effect=ValueError('invalid pair') if fail_pair else None), \
             patch('scripts.a2_stage1b_option2.subprocess.run',side_effect=command):
            done=run(plan,root/'out')
        self.assertTrue(done['execution_complete'])
        self.assertEqual(done['scores']['P300-parent'],parent)
        self.assertEqual(done['production_batch_size'],2)
        self.assertEqual(sum(m=='eval.a2_score' for m,a in events),5)
        self.assertEqual(sum(m=='scripts.verify_a2_stage1b_option2' for m,a in events),1)
        self.assertEqual(done['scientific_complete'],not fail_score and not fail_pair)
        old=json.loads((root/'out/batch_decision.json').read_text());self.assertFalse(old['hard_batch_passed'])
        self.assertTrue((root/'out/REPORT.md').is_file())
        self.assertEqual(done['models'],5 if fail_score else 6)

    def test_reuses_parent_and_fullset_failure_is_diagnostic(self):
        with tempfile.TemporaryDirectory() as d:self.exercise(Path(d))

    def test_one_failed_model_does_not_stop_other_models_or_replay(self):
        with tempfile.TemporaryDirectory() as d:self.exercise(Path(d),fail_score=True)

    def test_failed_paired_check_does_not_stop_independent_replay(self):
        with tempfile.TemporaryDirectory() as d:self.exercise(Path(d),fail_pair=True)

    def test_native_contract_records_completion_without_inventing_scientific_pass(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'jobs.json';write(p,job_config(dict(path='/plan',sha256='a'*64),continuation=True))
            job=load_jobs(p)[0]
            self.assertIsNone(job['timeout_seconds'])
            self.assertEqual(job['required_outputs'][0]['json_equals'],dict(execution_complete=True))
