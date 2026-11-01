import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from tests.a2.test_model import tiny_model
from efficient_llm.a2.model import KDModel
from efficient_llm.a2.train import HealTrainer,make_arguments
from efficient_llm.a2_depth.heal_short import ReportTrainer,Trace
from efficient_llm.a2_depth.common import write


class DepthShortTests(unittest.TestCase):
    def test_loss_gradient_recipe_unchanged(self):
        student=tiny_model();other=copy.deepcopy(student);teacher=copy.deepcopy(student).requires_grad_(False)
        x=object.__new__(HealTrainer);y=object.__new__(ReportTrainer)
        x.teacher=y.teacher=teacher;x.seen=[];y.seen=[]
        data=dict(input_ids=torch.tensor([[1,2,3,4]]),sample_id=torch.tensor([0]))
        a=x.compute_loss(KDModel(student,2),data);b=y.compute_loss(KDModel(other,2),data)
        self.assertTrue(torch.equal(a,b));a.backward();b.backward()
        for p,q in zip(student.parameters(),other.parameters()):
            if p.grad is not None:self.assertTrue(torch.equal(p.grad,q.grad))
        self.assertIs(ReportTrainer.create_optimizer,HealTrainer.create_optimizer)
        self.assertIs(ReportTrainer.create_scheduler,HealTrainer.create_scheduler)
        args=make_arguments('/tmp/depth-short-fixture',762,.001,4,8,cuda=False)
        self.assertEqual(args.gradient_accumulation_steps,2);self.assertFalse(args.gradient_checkpointing)
        self.assertFalse(hasattr(Trace(),'limit'))

    def test_dt4_failure_does_not_cancel_any_short_fit(self):
        import scripts.a2_depth_partb_run as runner
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg=root/'plan.json';write(cfg,dict(results_root=d));calls=[]
            def fake(cmd,**kwargs):
                calls.append(cmd);return SimpleNamespace(returncode=1 if '--dt4' in cmd else 0)
            with patch('sys.argv',['runner','--config',str(cfg),'--receipt',str(root/'done.json')]),patch.object(runner.subprocess,'run',side_effect=fake),patch('scripts.a2_depth_partb_report.finalize',return_value=dict(coverage_complete=False)):
                runner.main()
            self.assertEqual(len(calls),13)
            self.assertEqual(runner.registered_runs()[1:4],[('dclm',a,'W9',False) for a in ('P300','P1T','X72')])
            self.assertEqual(runner.registered_runs()[-3:],[('fwe',a,'W1',False) for a in ('P300','P1T','X72')])
            self.assertTrue(json.loads((root/'done.json').read_text())['execution_complete'])
            self.assertFalse(json.loads((root/'done.json').read_text())['coverage_complete'])

    def test_B_saved_arrays_absolute_X72_difference(self):
        from scripts.a2_depth_partb_report import compute
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg=dict(final_sequences=2,final_targets_per_sequence=1,short_S1_reference={ds:dict(E=.01,E_relative=.1,A_init=.04,A_init_relative=.2) for ds in ('dclm','fwe')})
            def row(value):return dict(per_sequence_nll=[value]*2,loss=value,sequences=2,tokens=2)
            dt4=root/'heals/ST1-dclm-P300-S1-0.2B-lr1e-3/result.json'
            write(dt4,dict(scores=dict(final=row(2.942554079))))
            for ds in ('dclm','fwe'):
                for block in ('W1','W9'):
                    for arm,base,post in [('P300',3.,3.2),('P1T',2.9,3.13),('X72',3.1,3.25)]:
                        name=f'D-{ds}-{arm}-{block}-0.2B-lr1e-3'
                        write(root/'heals'/name/'result.json',dict(baseline=row(base),scores=dict(final=row(post)),checkpoint='fixture',delta_post=post-base))
            r,tests,runs=compute(cfg,root)
            self.assertTrue(r['coverage_complete']);self.assertEqual(len(runs),12);self.assertTrue(all(t['passed'] for t in tests))
            self.assertAlmostEqual(r['datasets']['dclm']['W1']['E'],.03)
            self.assertAlmostEqual(r['datasets']['dclm']['W1']['A_init'],.05)
            self.assertAlmostEqual(r['datasets']['dclm']['W1']['A_init_relative'],.25)
            self.assertEqual(r['predictions'],dict(B1=True,B2=True))
            write(dt4,dict(scores=dict(final=dict(loss=2.942554079))))
            write(root/'coordinator_checks.json',dict(checks=[dict(test='process:DT4',passed=False)]))
            broken,_,_=compute(cfg,root)
            self.assertFalse(broken['coverage_complete'])
            self.assertTrue(any(f.get('test')=='process:DT4' for f in broken['flags']))
            self.assertEqual(broken['predictions'],r['predictions'])


if __name__=='__main__':unittest.main()
