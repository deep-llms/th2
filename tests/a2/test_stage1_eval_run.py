import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval.a2_run import write, lock
from eval.a2_request_bridge import digest
from eval.a2_stage1_run import run_screen, validate_plan
from eval.a2_stage1_stats import MODELS
from eval.downstream_suite import COUNTS


class Stage1EvalRunTest(unittest.TestCase):
    def fixture(self, root):
        assets=write(root/'assets.json',{})
        reference=write(root/'reference.json',{})
        selection={};identities={};rows={}
        for name in MODELS:
            arm=name.split('-')[0];healed=name.endswith('student')
            parent='S72' if arm=='X72' else arm
            selection[name]=dict(kind='healed' if healed else 'parent',parent=parent)
            identity=dict(schema=2 if healed else 1,dataset='dclm',parent=parent,
                          kind=selection[name]['kind'],assets=assets)
            if healed:
                run_id='ST1-'+arm
                selection[name]['run_id']=run_id
                rows[run_id]=dict(arm=arm,budget='3.2B',teacher='P300' if arm=='X72' else arm)
                identity['training_receipts']=dict(reference=reference,
                    result=write(root/(name+'-result.json'),rows[run_id]))
            identities[name]=write(root/(name+'.json'),identity)
        stage=write(root/'stage.json',{})
        bundle=write(root/'bundle.json',dict(schema=2,identities=identities,
            selection=selection,source_locks=dict(stage=stage,assets=assets)))
        plan=dict(schema=2,tasks=list(COUNTS),gpus=list(range(8)),max_length=4096,
            batch_size=2,native_python='/native/python',eval_python='/eval/python',
            tokenizer='/tokenizer',dataset_root='/benchmarks',
            manifest=write(root/'manifest.json',{}),input_spec=write(root/'input.json',{}),
            identities=identities,identity_bundle=bundle)
        return plan,({'reference':reference},{},rows,selection)

    def test_fixed_identity_bundle_and_batch_required(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plan,terminal=self.fixture(root)
            with patch('eval.a2_stage1_run.validate_terminal_bundle',return_value=terminal), \
                 patch('eval.a2_stage1_run.verify_identity'):
                self.assertEqual(set(validate_plan(plan)),set(MODELS))
                for change in ('batch','missing','identity'):
                    bad=copy.deepcopy(plan)
                    if change=='batch':bad['batch_size']=8
                    elif change=='missing':bad['identities'].pop('X72-student')
                    else:bad['identities']['P300-student']=bad['identities']['P1T-student']
                    with self.assertRaises(ValueError):validate_plan(bad)

    def exercise(self, root, fail_gate=False, fail_stats=False):
        plan,terminal=self.fixture(root);prepared=dict(schema=1,requests=[],protocol={});calls=[]
        def fake(argv,**kwargs):
            calls.append(argv);target=Path(argv[argv.index('--output')+1]);target.parent.mkdir(parents=True,exist_ok=True)
            if argv[2]=='eval.a2_requests':
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
                value=prepared if argv[3]=='prepare' else {}
            else:
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'0')
                identity=json.loads(Path(argv[argv.index('--identity')+1]).read_text())
                value=dict(passed=not fail_gate,production_batch_size=2,
                    checkpoint_identity_sha256=digest(identity),prepared_sha256=digest(prepared))
            target.write_text(json.dumps(value))
        def scores(jobs,**kwargs):
            self.assertEqual(len(jobs),6);self.assertEqual(len({j['gpu'] for j in jobs}),6)
            for job in jobs:
                self.assertEqual(job['argv'][job['argv'].index('--batch-size')+1],'2')
                target=Path(job['output']);target.parent.mkdir(parents=True);target.write_text('{}')
        with patch('eval.a2_stage1_run.validate_terminal_bundle',return_value=terminal), \
             patch('eval.a2_stage1_run.verify_identity'),patch('subprocess.run',side_effect=fake), \
             patch('eval.a2_run.execute_scoring',side_effect=scores) as scoring, \
             patch('eval.a2_run.collect_files',return_value={'status':'ok'}), \
             patch('eval.a2_stage1_run.aligned_accuracy_arrays',return_value={},
                   side_effect=ValueError('unpaired') if fail_stats else None), \
             patch('eval.a2_stage1_run.paired_accuracy_intervals',return_value={'passed':True}) as bootstrap:
            if fail_gate or fail_stats:
                with self.assertRaises(ValueError):run_screen(plan,root/'out',timeout_seconds=60)
                self.assertFalse((root/'out/complete.json').exists())
                if fail_gate:scoring.assert_not_called()
                bootstrap.assert_not_called()
            else:
                run_screen(plan,root/'out',timeout_seconds=60)
                bootstrap.assert_called_once_with({},resamples=10000,seed=0)
                self.assertEqual(sum(c[2]=='eval.a2_check' for c in calls),2)
                self.assertEqual(sum(c[2]=='eval.a2_requests' and c[3]=='replay' for c in calls),6)
                complete=json.loads((root/'out/complete.json').read_text())
                self.assertEqual(complete['models'],6)
                self.assertEqual(complete['additional_results']['paired_accuracy'],lock(root/'out/paired_accuracy.json'))

    def test_shared_requests_parent_and_student_gates_six_scores_and_paired_report(self):
        with tempfile.TemporaryDirectory() as d:self.exercise(Path(d))

    def test_failed_gate_prevents_scores(self):
        with tempfile.TemporaryDirectory() as d:self.exercise(Path(d),fail_gate=True)

    def test_unpaired_report_prevents_complete_marker(self):
        with tempfile.TemporaryDirectory() as d:self.exercise(Path(d),fail_stats=True)


if __name__=='__main__':unittest.main()
