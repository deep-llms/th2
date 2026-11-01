import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from transformers import default_data_collator,set_seed
from tests.a2.test_model import tiny_model
from tests.a2.test_depth_resume import RandomProbe
from efficient_llm.a2.model import KDModel,prune
from efficient_llm.a2.train import make_arguments
from efficient_llm.a2_depth.heal_short import ReportTrainer,Trace
from efficient_llm.a2_depth.heal_long import train_resumable
from efficient_llm.a2_depth.resume import enable_checkpoints,select_checkpoint
from efficient_llm.a2_depth.common import read,write,SPEC_SHA

class DepthLongTests(unittest.TestCase):
    def test_integrated_resume_and_completed_checkpoint_no_extra_update(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);teacher=tiny_model();student=copy.deepcopy(teacher);prune(student,[1])
            rows=[dict(input_ids=torch.tensor([1,2+i%20,3,4,5]),sample_id=i) for i in range(32)]
            identity=dict(steps=16,global_sequences=2,microbatch=1,lr=.001)
            def run(out,stop=None,resume=False):
                set_seed(0);trace=Trace()
                args=enable_checkpoints(make_arguments(out/'trainer',16,.001,1,1,cuda=False,global_sequences=2),every=4)
                trainer=ReportTrainer(model=KDModel(copy.deepcopy(student),2),teacher=copy.deepcopy(teacher),args=args,
                    train_dataset=rows,data_collator=default_data_collator,callbacks=[trace,RandomProbe(stop)])
                ckpt,rejected=select_checkpoint(out/'trainer',identity,1) if resume else (None,[])
                if ckpt and read(ckpt/'trainer_state.json')['global_step']==16:
                    with patch.object(trainer,'train',side_effect=AssertionError('Extra optimizer update')):
                        train_resumable(trainer,trace,identity,ckpt,out,resume=resume,rejected=rejected)
                else:train_resumable(trainer,trace,identity,ckpt,out,resume=resume,rejected=rejected)
                return trainer,trace
            full,ft=run(root/'full');run(root/'interrupted',stop=4)
            resumed,rt=run(root/'interrupted',resume=True)
            final,tt=run(root/'interrupted',resume=True)
            for value in (resumed,final):
                self.assertEqual(full.seen,value.seen);self.assertEqual(value.state.global_step,16)
                for k,v in full.model.state_dict().items():torch.testing.assert_close(v,value.model.state_dict()[k],rtol=0,atol=0)
            self.assertEqual(ft.lrs,rt.lrs);self.assertEqual(ft.lrs,tt.lrs)
            self.assertEqual(len(read(root/'interrupted/resume_events.json')['events']),3)

    def test_partc_labels_ratios_and_incomplete_notes(self):
        from scripts.a2_depth_partc_report import finalize
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg=dict(results_root=d,final_sequences=2,final_targets_per_sequence=1,
                stage1_dclm_reference={'3.2B':dict(E=.02,A_init=.012)})
            def row(v):return dict(loss=v,per_sequence_nll=[v,v],sequences=2,tokens=2)
            for arm,parent,val in [('P300',3.,3.1),('P1T',2.9,3.01),('X72',3.2,3.101)]:
                write(root/'heals'/f'ST1-dclm-{arm}-S1-12.8B-lr1e-3/result.json',
                      dict(scores=dict(final=row(val)),baseline=row(parent),delta_post=val-parent,checkpoint='fixture'))
                result=finalize(cfg,note_arm=arm)
                if arm!='X72':self.assertFalse(result['coverage_complete'])
            self.assertTrue(result['coverage_complete']);self.assertAlmostEqual(result['E'],.01)
            self.assertAlmostEqual(result['A_init'],.001);self.assertAlmostEqual(result['E_ratio_3_2B'],.5)
            self.assertEqual(result['E_label']['label'],'persists');self.assertEqual(result['A_init_label']['label'],'no effect')
            self.assertEqual(result['predictions'],dict(C1=True,C2=True))
            self.assertTrue(any(f.get('test')=='DT6_evidence' for f in result['flags']))
            self.assertEqual(read(root/'notes/X72.json')['spec_sha256'],SPEC_SHA)

class DepthLongCoordinatorTests(unittest.TestCase):
    def test_interrupted_arm_resumes_advancing_state_then_continues(self):
        from types import SimpleNamespace
        import scripts.a2_depth_partc_run as runner
        from efficient_llm.a2_depth.common import Checks
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'heals/ST1-dclm-P300-S1-12.8B-lr1e-3';out.mkdir(parents=True)
            cfg=dict(results_root=d,spec_sha256=SPEC_SHA,assets={},heal_streams={'dclm':{}},source_manifest={})
            calls=[]
            def fake(cmd,**kw):calls.append(cmd);return SimpleNamespace(returncode=1 if len(calls)<3 else 0)
            checks=Checks(root/'checks.json')
            with patch.object(runner,'require_free_gpus') as guard,patch.object(runner.subprocess,'run',side_effect=fake),patch.object(runner,'select_checkpoint',side_effect=[(out/'trainer/checkpoint-1000',[]),(out/'trainer/checkpoint-2000',[])]):
                events=runner.run_arm(cfg,'fixture','P300',checks)
            self.assertEqual(len(events),3);self.assertEqual(guard.call_count,3)
            self.assertTrue(all('--resume' in call for call in calls))
            self.assertTrue(any(not c['passed'] for c in checks.items))
            with patch.object(runner,'require_free_gpus',side_effect=RuntimeError('unknown GPU')),patch.object(runner.subprocess,'run') as proc:
                with self.assertRaises(RuntimeError):runner.run_arm(cfg,'fixture','P1T',checks)
                proc.assert_not_called()

    def test_dt6_flag_does_not_cancel_three_arms_or_their_notes(self):
        import scripts.a2_depth_partc_run as runner
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plan=root/'config.json'
            write(plan,dict(results_root=d,heal_streams={'dclm':{'path':'fixture'}},stream_history={}))
            arms=[];notes=[]
            def run_arm(cfg,config,arm,checks):
                arms.append(arm);return [dict(arm=arm,returncode=0)]
            def finalize(cfg,**kw):
                notes.append(kw.get('note_arm'));return dict(coverage_complete=True)
            with patch('sys.argv',['runner','--config',str(plan),'--receipt',str(root/'done.json')]),patch('efficient_llm.a2_depth.stream.verify',side_effect=ValueError('DT6 fixture flag')),patch.object(runner,'run_arm',side_effect=run_arm),patch('scripts.a2_depth_partc_report.finalize',side_effect=finalize):
                runner.main()
            self.assertEqual(arms,['P300','P1T','X72']);self.assertEqual(notes,['P300','P1T','X72',None])
            self.assertFalse(read(root/'coordinator_checks.json')['checks'][0]['passed'])
            self.assertFalse(read(root/'done.json')['downstream_complete'])

    def test_repeated_same_step_failure_is_flagged_without_endless_retry(self):
        from types import SimpleNamespace
        import scripts.a2_depth_partc_run as runner
        from efficient_llm.a2_depth.common import Checks
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg=dict(results_root=d,spec_sha256=SPEC_SHA,assets={},heal_streams={'dclm':{}},source_manifest={})
            checks=Checks(root/'checks.json')
            with patch.object(runner,'require_free_gpus'),patch.object(runner.subprocess,'run',return_value=SimpleNamespace(returncode=1)),patch.object(runner,'select_checkpoint',return_value=(root/'checkpoint-1000',[])):
                events=runner.run_arm(cfg,'fixture','P300',checks)
            self.assertEqual(len(events),2)
            self.assertEqual(checks.items[-1]['test'],'resume_no_progress:P300')
            self.assertFalse(checks.items[-1]['passed'])


if __name__=='__main__':unittest.main()
