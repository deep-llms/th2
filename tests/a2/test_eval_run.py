import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eval.a2_run import run_screen, validate_plan, write, lock
from eval.a2_request_bridge import digest
from eval.downstream_suite import COUNTS


class EvalRunTests(unittest.TestCase):
    def fixture(self,root):
        plan=dict(schema=1,tasks=list(COUNTS),gpus=list(range(8)),max_length=4096,
            native_python='/native/python',eval_python='/eval/python',tokenizer='/tokenizer',
            dataset_root='/benchmarks',manifest=write(root/'manifest.json',{}),
            input_spec=write(root/'input.json',{}),identities={})
        for ds in ('dclm','fwe','nem'):
            for parent in ('P300','P1T'):
                for budget in [None]+(['0.2B','0.8B'] if ds=='nem' else ['0.2B']):
                    name=f'{ds}-{parent}-'+('original' if budget is None else f'S1-{budget}')
                    identity=dict(dataset=ds,parent=parent,kind='parent' if budget is None else 'healed')
                    if budget:
                        identity['training_receipts']=dict(result=write(root/(name+'-result.json'),
                            dict(set='S1',budget=budget,teacher=parent)))
                    plan['identities'][name]=write(root/(name+'.json'),identity)
        return plan

    def test_plan_rejects_missing_checkpoint_or_wrong_budget(self):
        with tempfile.TemporaryDirectory() as d,patch('eval.a2_run.verify_identity'):
            root=Path(d);plan=self.fixture(root)
            self.assertEqual(len(validate_plan(plan)),14)
            bad=copy.deepcopy(plan);bad['identities'].pop('nem-P1T-S1-0.8B')
            with self.assertRaises(ValueError):validate_plan(bad)
            item=plan['identities']['nem-P1T-S1-0.8B'];identity=json.loads(Path(item['path']).read_text())
            path=Path(identity['training_receipts']['result']['path']);path.write_text(json.dumps(dict(set='S2',budget='0.8B',teacher='P1T')))
            identity['training_receipts']['result']=lock(path)
            Path(item['path']).write_text(json.dumps(identity));plan['identities']['nem-P1T-S1-0.8B']=lock(item['path'])
            with self.assertRaises(ValueError):validate_plan(plan)

    def exercise(self,root,fail=False):
        plan=self.fixture(root);prepared=dict(schema=1,requests=[],protocol={});calls=[]
        def fake(argv,**kwargs):
            calls.append(argv);out=Path(argv[argv.index('--output')+1]);out.parent.mkdir(parents=True,exist_ok=True)
            if argv[2]=='eval.a2_requests':
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
                value=prepared if argv[3]=='prepare' else {}
            else:
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'0')
                identity=json.loads(Path(argv[argv.index('--identity')+1]).read_text())
                value=dict(passed=not fail,checkpoint_identity_sha256=digest(identity),prepared_sha256=digest(prepared))
            out.write_text(json.dumps(value))
        def scores(jobs,**kwargs):
            self.assertEqual(len(jobs),14)
            self.assertEqual(len({j['argv'][j['argv'].index('--prepared')+1] for j in jobs}),1)
            for job in jobs:
                out=Path(job['output']);out.parent.mkdir(parents=True);out.write_text('{}')
        with patch('eval.a2_run.verify_identity'),patch('subprocess.run',side_effect=fake), \
             patch('eval.a2_run.execute_scoring',side_effect=scores) as scoring, \
             patch('eval.a2_run.collect_files',return_value={'status':'ok'}) as collect:
            if fail:
                with self.assertRaises(ValueError):run_screen(plan,root/'out',timeout_seconds=60)
                scoring.assert_not_called();collect.assert_not_called()
                self.assertFalse((root/'out/complete.json').exists())
            else:
                run_screen(plan,root/'out',timeout_seconds=60)
                self.assertEqual(sum(c[2]=='eval.a2_check' for c in calls),2)
                self.assertEqual(sum(c[2]=='eval.a2_requests' and c[3]=='prepare' for c in calls),1)
                self.assertEqual(sum(c[2]=='eval.a2_requests' and c[3]=='replay' for c in calls),14)
                self.assertEqual(json.loads((root/'out/complete.json').read_text())['models'],14)

    def test_shared_requests_two_gates_then_fourteen_models(self):
        with tempfile.TemporaryDirectory() as d:self.exercise(Path(d))

    def test_failed_gpu_gate_never_starts_scoring(self):
        with tempfile.TemporaryDirectory() as d:self.exercise(Path(d),fail=True)

    def test_scoring_deadline_shared_across_sequential_jobs(self):
        from eval.a2_suite import execute_scoring
        with tempfile.TemporaryDirectory() as d:
            jobs=[dict(gpu=0,output=str(Path(d)/str(i)/'score.json'),argv=[str(i)]) for i in range(2)]
            def fake(argv,**kwargs):
                self.assertEqual(kwargs['timeout'],9)
                Path(jobs[int(argv[0])]['output']).write_text('{}')
            with patch('time.monotonic',side_effect=[0,1,11]),patch('subprocess.run',side_effect=fake) as call:
                with self.assertRaises(TimeoutError):execute_scoring(jobs,timeout_seconds=10)
                self.assertEqual(call.call_count,1)


if __name__=='__main__':unittest.main()
