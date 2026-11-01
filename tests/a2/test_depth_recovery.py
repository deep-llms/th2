import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from efficient_llm.a2_depth.common import read,write,Checks,SPEC_SHA
from efficient_llm.a2_depth.resume import checkpoint_files,select_checkpoint,pin_clean_checkpoint,finite_state
from efficient_llm.a2_depth.recovery import seal_arm,completed_arm,continued_checks,coordinator_lock
from efficient_llm.a2.protocol import lr_for_update
from scripts.a2_quant_io import lock


class RecoveryTests(unittest.TestCase):
    def checkpoint(self,trainer,step,bad=0):
        identity=dict(steps=16,lr=.001,global_sequences=2,microbatch=1)
        root=trainer/f'checkpoint-{step}';root.mkdir(parents=True)
        for name in checkpoint_files(1):(root/name).write_bytes(b'fixture')
        write(root/'trainer_state.json',dict(global_step=step))
        write(root/'depth-audit-rank0.json',dict(identity=identity,global_step=step,world_size=1,rank=0,
            lrs=[lr_for_update(i,16,.001) for i in range(1,step+1)],seen=[[i] for i in range(2*step)],nonfinite_microbatches=bad))
        write(root/'depth-checkpoint.json',dict(complete=True,identity=identity,global_step=step,world_size=1,
            files={n:(root/n).stat().st_size for n in checkpoint_files(1)}))
        return root,identity

    def test_clean_copy_survives_rotation_and_rejects_nonfinite_state(self):
        import shutil
        with tempfile.TemporaryDirectory() as d:
            trainer=Path(d)/'trainer';clean,identity=self.checkpoint(trainer,2)
            pin_clean_checkpoint(clean);pinned=trainer/'clean-state/checkpoint-2'
            (clean/'optimizer.pt').write_bytes(b'changed')
            self.assertEqual((pinned/'optimizer.pt').read_bytes(),b'fixture')
            shutil.rmtree(clean)  # Native rotation must not delete pinned state.
            bad,_=self.checkpoint(trainer,4,bad=1)
            selected,rejected=select_checkpoint(trainer,identity,1)
            self.assertEqual(selected,pinned);self.assertEqual(len(rejected),1)
            self.assertIn('Nonfinite loss',rejected[0]['error'])
            newer,_=self.checkpoint(trainer,6)
            m=read(newer/'depth-checkpoint.json');m['state_finite']=False;write(newer/'depth-checkpoint.json',m)
            self.assertEqual(select_checkpoint(trainer,identity,1)[0],pinned)
            self.assertFalse(finite_state({'state':[torch.tensor([float('nan')])]}))
            self.assertTrue(finite_state({'state':[torch.tensor([1.])]}))
            m['state_finite']=True;write(newer/'depth-checkpoint.json',m);pin_clean_checkpoint(newer)
            self.assertFalse(pinned.exists());self.assertEqual(select_checkpoint(trainer,identity,1)[0],newer)

    def test_nonfinite_is_flagged_before_backward_and_finite_gradient_unchanged(self):
        from efficient_llm.a2_depth.heal_long import RecoverableTrainer
        with tempfile.TemporaryDirectory() as d:
            trainer=object.__new__(RecoverableTrainer);trainer.state=SimpleNamespace(global_step=7)
            trainer.recovery_checks=Checks(Path(d)/'checks.json')
            w=torch.tensor(2.,requires_grad=True);loss=w*w
            with patch('efficient_llm.a2_depth.heal_short.ReportTrainer.compute_loss',return_value=loss):
                out=trainer.compute_loss(None,{});self.assertIs(out,loss);out.backward()
            self.assertEqual(w.grad.item(),4.)
            w.grad=None
            with patch('efficient_llm.a2_depth.heal_short.ReportTrainer.compute_loss',return_value=w*float('nan')):
                with self.assertRaises(FloatingPointError):trainer.compute_loss(None,{})
            self.assertIsNone(w.grad);self.assertEqual(w.item(),2.)
            self.assertFalse(read(Path(d)/'checks.json')['checks'][0]['passed'])

    def test_third_attempt_recovers_same_step_and_preserves_prior_failures(self):
        import scripts.a2_depth_partc_run as runner
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg=dict(results_root=d,spec_sha256=SPEC_SHA,assets={},heal_streams={'dclm':{}},source_manifest={})
            checks=Checks(root/'checks.json');checks.record('old_failure',False,'retained')
            with patch.object(runner,'completed_arm',return_value={}),patch.object(runner,'require_free_gpus'),patch.object(runner.time,'sleep') as sleep,patch.object(runner.subprocess,'run',side_effect=[SimpleNamespace(returncode=1),SimpleNamespace(returncode=1),SimpleNamespace(returncode=0)]) as proc,patch.object(runner,'select_checkpoint',return_value=(root/'checkpoint-1000',[])):
                events=runner.run_arm(cfg,'config','P300',continued_checks(root/'checks.json'))
            self.assertEqual(len(events),3);self.assertEqual(len(read(root/'attempts.json')['events']),3)
            self.assertEqual([x.args[0] for x in sleep.call_args_list],[10,20])
            self.assertFalse(read(root/'checks.json')['checks'][0]['passed'])
            self.assertEqual(proc.call_count,3)

    def test_completed_arm_skips_compute_and_detects_tampered_weights(self):
        import scripts.a2_depth_partc_run as runner
        from efficient_llm.a2_depth.heal_long import run_identity
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg=dict(results_root=d,spec_sha256=SPEC_SHA,assets={},heal_streams={'dclm':{}},source_manifest={})
            identity=run_identity(cfg,'P300');out=root/'heals/ST1-dclm-P300-S1-12.8B-lr1e-3'
            write(out/'result.json',dict(status='complete',full_state_identity=identity,lr_updates=[0.]*48828))
            write(out/'recipe.json',{});write(out/'order.json',{})
            for rank in range(8):write(out/f'tests-rank{rank}.json',dict(checks=[]))
            (out/'final').mkdir();(out/'final/model.safetensors').write_bytes(b'weights')
            seal_arm(out,identity)
            with patch.object(runner.subprocess,'run') as proc:
                events=runner.run_arm(cfg,'config','P300',Checks(root/'checks.json'))
            proc.assert_not_called();self.assertEqual(events[0]['action'],'reuse verified completed arm')
            (out/'final/model.safetensors').write_bytes(b'changed')
            with self.assertRaises(ValueError):completed_arm(out,identity)
            with coordinator_lock(root):
                with self.assertRaises(BlockingIOError):
                    with coordinator_lock(root):pass

    def test_downstream_reentry_reuses_sealed_scores_and_replays(self):
        import scripts.a2_depth_downstream as downstream
        from eval.a2_request_bridge import digest
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);prepared=root/'prepared.json';write(prepared,dict(requests=[1,2]))
            cfg=dict(results_root=d,spec_sha256=SPEC_SHA,source_manifest={},prepared=lock(prepared),prior_downstream={'scores':{},'replays':{}},downstream_common={'eval_python':'fixture','tokenizer':'fixture','dataset_root':'fixture','tasks':[]})
            cfg['downstream_common'].update(input_spec=lock(prepared),manifest=lock(prepared))
            for arm in ('P300','P1T','S72'):
                name=arm+'-parent';p=root/(name+'.json');write(p,dict(batch_size=2,checkpoint_identity={'arm':arm}))
                cfg['prior_downstream']['scores'][name]=lock(p);cfg['prior_downstream']['replays'][name]=lock(p)
            calls=[]
            def fake(args,**kw):
                calls.append(args);out=Path(args[args.index('--output')+1])
                if 'scripts.a2_depth_score' in args:
                    arm=args[args.index('--arm')+1];ident={'arm':arm}
                    item=write(out/'scores.json',dict(batch_size=2,scores=[1,2],prepared_sha256=digest(read(prepared)),checkpoint_identity=ident,checkpoint_identity_sha256=digest(ident)))
                    write(out/'checks.json',dict(checks=[]));write(out/'complete.json',dict(coverage_complete=True,scores=item,checks=[]))
                else:write(out,dict(replay=True))
                return SimpleNamespace(returncode=0)
            with patch('scripts.a2_depth_partc_run.require_free_gpus'),patch.object(downstream.subprocess,'run',side_effect=fake),patch('scripts.a2_depth_score.identity',side_effect=lambda cfg,arm,checks:{'arm':arm}),patch.object(downstream,'collect_files',return_value={}),patch.object(downstream,'paired',return_value={'ci':[0,1]}),patch.object(downstream,'verify_replay',return_value=True):
                first=downstream.run(cfg,'config');count=len(calls)
                second=downstream.run(cfg,'config')
                self.assertTrue(first['coverage_complete'] and second['coverage_complete']);self.assertEqual(len(calls),count)
                # A partial/corrupt score is preserved, only this arm is re-scored.
                original=root/'downstream/scores/P1T-student/scores.json';original.write_text('{}')
                third=downstream.run(cfg,'config')
                self.assertTrue(third['coverage_complete']);self.assertEqual(len(calls),count+1)
                self.assertEqual(original.read_text(),'{}')
                self.assertTrue((root/'downstream/scores/P1T-student-attempt001/scores.json').exists())

class ContinuationTests(unittest.TestCase):
    def fixture(self,root):
        import scripts.a2_depth_continue as continuation
        out=root/'experiment';frozen=out/'source';frozen.mkdir(parents=True)
        files={}
        for rel in ('scripts/a2_depth_continue.py','scripts/launch_a2_depth_partc.py'):
            p=frozen/rel;p.parent.mkdir(exist_ok=True);p.write_bytes(Path(rel).read_bytes());files[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
        plan=frozen/'plan.json';write(plan,{'registered':'fixture'})
        source=write(out/'source-manifest.json',dict(code_commit='frozen',files=files))
        resultroot=out/'results/a2_depth_20261031';resultroot.mkdir(parents=True)
        config=write(resultroot/'config/plan.json',dict(code_commit='frozen',results_root=str(resultroot),source_manifest=source,predecessor={'burn_script':'known'}))
        from scripts.a2_quant_io import write as write_native
        from scripts.launch_a2_depth_partc import job_config
        jobs=write_native(out/'jobs.json',job_config(config));live={'host':'thiennh-p6-6y3o-worker-0','gpus':[{'index':i,'uuid':str(i),'pids':[]} for i in range(8)]}
        inspection=write(out/'inspection.json',live)
        write(out/'launch.json',dict(config=config,input_plan=lock(plan),source_manifest=source,jobs=jobs,session='original',initial_inspection=inspection,environment={'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(frozen)}))
        handoff=root/'handoff.json';write(handoff,dict(mode='free'))
        guard=root/'guard/DISABLED';guard.parent.mkdir();(guard.parent/'gpu_guard.sh').write_bytes(b'guard')
        args=SimpleNamespace(output=out,plan=plan,code_commit='frozen',session='continue01',recovery_handoff=handoff)
        return args,live,guard

    def test_restart_keeps_frozen_plan_results_and_uses_new_native_supervisor(self):
        import scripts.a2_depth_continue as continuation
        with tempfile.TemporaryDirectory() as d:
            args,live,guard=self.fixture(Path(d));calls=[]
            def process(argv,**kw):
                calls.append((argv,kw));return SimpleNamespace(returncode=1 if argv[:2]==['tmux','has-session'] else 0,stdout='1\n')
            with patch.object(continuation,'GUARD',guard),patch.object(continuation,'GUARD_HASH',hashlib.sha256(b'guard').hexdigest()),patch.object(continuation,'inspect',return_value=live),patch.object(continuation.socket,'gethostname',return_value=live['host']),patch.object(continuation.subprocess,'run',side_effect=process):
                continuation.continue_run(args)
                record=read(args.output/'continuations.json')['attempts'][0]
                self.assertEqual(record['session'],'continue01')
                launch=next(v for v in calls if v[0][:2]==['tmux','new-session'])
                self.assertIn('continuations/attempt001/supervised',launch[0][-1])
                self.assertEqual(launch[0][launch[0].index('-c')+1],str(args.output/'source'))
                self.assertEqual(read(record['config']['path'])['code_commit'],'frozen')
                # A still-held native guard or occupied GPU must never be bypassed.
                guard.write_text('owner')
                with self.assertRaises(ValueError):continuation.admission(args.output,read(args.output/'launch.json'),read(record['config']['path']),{'mode':'free'})
                guard.unlink();live['gpus'][0]['pids']=[123]
                with self.assertRaises(ValueError):continuation.admission(args.output,read(args.output/'launch.json'),read(record['config']['path']),{'mode':'free'})
            record=read(args.output/'launch.json')
            Path(record['config']['path']).write_text('{}')
            with self.assertRaises(ValueError):continuation.validate_existing(args.output,args.plan,'frozen')

    def test_changed_frozen_source_and_live_supervisor_refused(self):
        import scripts.a2_depth_continue as continuation
        with tempfile.TemporaryDirectory() as d:
            args,live,guard=self.fixture(Path(d))
            with patch.object(continuation.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='0\n')):
                with self.assertRaises(ValueError):continuation.continue_run(args)
            (args.output/'source/scripts/a2_depth_continue.py').write_text('changed')
            with self.assertRaises(ValueError):continuation.validate_existing(args.output,args.plan,'frozen')
