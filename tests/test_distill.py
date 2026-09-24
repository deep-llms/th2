import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pcc.distill_model import CorrectionStudent, FeedbackOff
from pcc.distill_training import digest, optimizer, restore, train, update, wrap
from pcc.joint_training import evaluate
from test_joint import contexts, settings, tiny


def student(arm='PCC', checkpoint_layers=True):
    teacher=tiny('Deep',checkpoint_layers=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        torch.nn.init.normal_(teacher.adapter.out.weight,std=.05)
    return CorrectionStudent(teacher,arm,.2,checkpoint_layers=checkpoint_layers)


def identity(model, cfg=None, world=1):
    return {'settings':asdict(cfg or replace(settings(),version='distill-v1')),
            'arm':model.arm,'world_size':world,'parent':'test-frozen-teacher'}


def ddp_worker(rank, directory):
    root=Path(directory);torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+str(root/'rendezvous'),rank=rank,world_size=2)
    try:
        cfg=replace(settings(),version='distill-v1',tokens_per_update=64,dev_tokens=35)
        data,dev=contexts(32),contexts(5);dev.valid[-1,3:]=False
        for arm in ('LM','PCC'):
            a,b=student(arm),student(arm)
            frozen=digest(b.teacher)
            def opt(m):
                return torch.optim.SGD([{'params':list(m.student.parameters()),'lr':.01,'peak_lr':.01}])
            oa,ob,parallel=opt(a),opt(b),wrap(b)
            for step in (1,2):
                ra=update(a,oa,data,step,cfg,1,mixed_precision=False)
                rb=update(b,ob,data,step,cfg,1,parallel=parallel,mixed_precision=False)
                assert abs(ra['nll']-rb['nll'])<1e-6
                assert abs(ra['correction_loss']-rb['correction_loss'])<1e-6
                for p,q in zip(a.student.parameters(),b.student.parameters()):
                    torch.testing.assert_close(p,q,atol=2e-7,rtol=1e-5)
            assert digest(b.teacher)==frozen
            assert all(p.grad is None and not p.requires_grad for p in b.teacher.parameters())
            from pcc.distributed import identical_parameters
            identical_parameters(b.student)
            x=evaluate(b,dev,1,mixed_precision=False)
            y=evaluate(b,dev,1,mixed_precision=False,distributed=True)
            np.testing.assert_array_equal(x[0],y[0]);np.testing.assert_array_equal(x[1],y[1])
            del parallel
        from pcc.distill import calibration
        from pcc.training import prefix_batches
        calibrated=calibration(b,data,budget=35)
        square,elements=0.,0
        for context in prefix_batches(data,35,1,'cpu'):
            with torch.autocast('cpu',dtype=torch.bfloat16):
                delta=b.teacher_correction(context)
            values=delta[:,:-1][context.targets()].double()
            square+=float(values.square().sum());elements+=values.numel()
        assert abs(calibrated['sigma_delta']-(square/elements)**.5)<1e-10
        assert calibrated['input_tokens']==35
        ids=identity(b,cfg,world=2)
        train(student(),data,dev,root/'full',ids,mixed_precision=False,distributed=True)
        train(student(),data,dev,root/'part',ids,mixed_precision=False,distributed=True,stop_after=2)
        train(student(),data,dev,root/'resume',ids,mixed_precision=False,distributed=True,resume=root/'part/latest.pt')
        full=torch.load(root/'full/final.pt',weights_only=True)
        resumed=torch.load(root/'resume/final.pt',weights_only=True)
        for name,tensor in full['student'].items():
            torch.testing.assert_close(tensor,resumed['student'][name],atol=0,rtol=0)
        if rank==0:(root/'passed.json').write_text('{"status":"ok"}')
    finally:
        dist.destroy_process_group()


class DistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_zero_student_matches_trained_backbone_with_feedback_off(self):
        model=student();context=contexts(2).batch(0,2,'cpu')
        for bf16 in (False,True):
            with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16,enabled=bf16):
                torch.testing.assert_close(model(context),FeedbackOff(model.teacher)(context),atol=0,rtol=0)
        self.assertEqual(digest(model.student),digest(student('LM').student))

    def test_teacher_correction_is_exact_same_layer_detached_target(self):
        model=student();context=contexts(2).batch(0,2,'cpu');actual=[]
        hook=model.teacher.adapter.register_forward_hook(lambda module,args,output:actual.append(output))
        with torch.no_grad():model.teacher(context)
        hook.remove()
        target=model.teacher_correction(context)
        torch.testing.assert_close(target,actual[0],atol=0,rtol=0)
        self.assertFalse(target.requires_grad)
        torch.testing.assert_close(target[:,0],torch.zeros_like(target[:,0]),atol=0,rtol=0)

    def test_inference_uses_each_backbone_block_once_and_no_teacher_adapter(self):
        model=student();context=contexts(2).batch(0,2,'cpu');calls=[0,0,0];teacher_calls=[]
        handles=[]
        for i,layer in enumerate(model.model.model.layers):
            def hook(module,args,result,i=i):calls[i]+=1
            handles.append(layer.register_forward_hook(hook))
        handles.append(model.teacher.adapter.register_forward_hook(lambda *args:teacher_calls.append(1)))
        with torch.no_grad():model(context)
        for h in handles:h.remove()
        self.assertEqual(calls,[1,1,1]);self.assertEqual(teacher_calls,[])

    def test_real_updates_freeze_teacher_and_preserve_causality(self):
        data=contexts()
        for arm in ('LM','PCC'):
            model=student(arm);before=digest(model.teacher);initial=digest(model.student)
            opt=optimizer(model)
            for step in (1,2):
                record=update(model,opt,data,step,settings(),1,mixed_precision=False)
            self.assertEqual(digest(model.teacher),before)
            self.assertNotEqual(digest(model.student),initial)
            self.assertFalse(model.teacher.training)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.teacher.parameters()))
            self.assertEqual(record['lambda_corr'],float(arm=='PCC'))
            context=data.batch(0,1,'cpu');changed=copy.deepcopy(context)
            changed.input_ids[:,5:]=(changed.input_ids[:,5:]+3)%67
            with torch.no_grad():
                torch.testing.assert_close(model(context)[:,:5],model(changed)[:,:5],atol=0,rtol=0)

    def test_checkpointing_and_microbatch_accumulation_match(self):
        data=contexts()
        for arm in ('LM','PCC'):
            a,b=student(arm,False),student(arm,True)
            def opt(m):return torch.optim.SGD([{'params':list(m.student.parameters()),'lr':.01,'peak_lr':.01}])
            oa,ob=opt(a),opt(b)
            for step in (1,2):
                update(a,oa,data,step,settings(),2,mixed_precision=False)
                update(b,ob,data,step,settings(),1,mixed_precision=False)
            for p,q in zip(a.student.parameters(),b.student.parameters()):
                torch.testing.assert_close(p,q,atol=2e-7,rtol=1e-5)

    def test_adapter_only_checkpoint_resume_and_identity_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data,dev=contexts(),contexts(4)
            ids=identity(student())
            train(student(),data,dev,root/'full',ids,mixed_precision=False)
            train(student(),data,dev,root/'part',ids,mixed_precision=False,stop_after=2)
            train(student(),data,dev,root/'resumed',ids,mixed_precision=False,resume=root/'part/latest.pt')
            full=torch.load(root/'full/final.pt',weights_only=True)
            resumed=torch.load(root/'resumed/final.pt',weights_only=True)
            self.assertNotIn('teacher',full);self.assertNotIn('model',full)
            for name,tensor in full['student'].items():
                torch.testing.assert_close(tensor,resumed['student'][name],atol=0,rtol=0)
            model=student();wrong={**ids,'parent':'another-checkpoint'}
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                restore(root/'part/latest.pt',model,optimizer(model),wrong)

    def test_two_process_distributed_gradients_calibration_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(ddp_worker,args=(directory,),nprocs=2,join=True)
            self.assertEqual(json.loads((Path(directory)/'passed.json').read_text())['status'],'ok')

    def test_manifest_uses_four_sequential_eight_gpu_runs(self):
        from argparse import Namespace
        from pcc.distill import manifest,settings as distill_settings
        from run_experiments import load_jobs
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args=Namespace(config=root/'config.json',joint_root=root/'joint',data_dir=root/'inputs',
                           output=root/'jobs.json',audit_root=root/'audits')
            manifest(args);jobs=load_jobs(args.output)
            self.assertEqual([j['name'] for j in jobs],['seed-0-LM','seed-0-PCC','seed-1-LM','seed-1-PCC','report'])
            for job in jobs[:4]:self.assertEqual(job['gpus'],list(range(8)))
            self.assertNotIn('gpus',jobs[-1])
            self.assertEqual(distill_settings({'experiment':'joint-v2-ddp'}).updates,6144)
            with self.assertRaises(ValueError):distill_settings({})


if __name__=='__main__':unittest.main()
