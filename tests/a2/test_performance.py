import copy
from pathlib import Path
import tempfile
import unittest
import torch
from transformers import default_data_collator
from efficient_llm.a2.model import KDModel,prune
from efficient_llm.a2.train import HealTrainer,make_arguments,audit_order
from efficient_llm.a2.performance import Options,PerformanceKDModel,PerformanceTrainer,VARIANTS
from test_model import tiny_model
from test_train import Rows


class PerformanceTests(unittest.TestCase):
    def test_chunk_and_compilation_loss_gradient(self):
        teacher=tiny_model().requires_grad_(False)
        student=copy.deepcopy(teacher).requires_grad_(True);prune(student,[1,2])
        ids=Rows(2,length=9)[0]['input_ids'][None]
        h=teacher.model(ids).last_hidden_state.detach();w=teacher.lm_head.weight
        ref=KDModel(student,3)(ids,h,w)['loss']
        grads=torch.autograd.grad(ref,tuple(student.parameters()))
        for compile_head in (False,True):
            # AOT eager tests capture/backward; real Inductor must pass B200 gate.
            model=PerformanceKDModel(student,Options(compile_head=compile_head),compile_backend='aot_eager')
            loss=model(ids,h,w)['loss'];got=torch.autograd.grad(loss,tuple(student.parameters()))
            torch.testing.assert_close(ref,loss,atol=1e-6,rtol=1e-5)
            for a,b in zip(grads,got):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-4)

    def test_deferred_audit_and_optimizer_updates(self):
        teacher=tiny_model().requires_grad_(False)
        reference=copy.deepcopy(teacher).requires_grad_(True);prune(reference,[1,2])
        states=[]
        with tempfile.TemporaryDirectory() as d:
            for i,option in enumerate((Options(),Options(microbatch=2,deferred_checks=True),Options(microbatch=4,deferred_checks=True,fused_optimizer=True))):
                student=copy.deepcopy(reference)
                args=make_arguments(Path(d)/str(i),12,1e-4,option.microbatch,1,cuda=False,global_sequences=8)
                if i==0:
                    trainer=HealTrainer(model=KDModel(student),teacher=teacher,args=args,
                        train_dataset=Rows(96),data_collator=default_data_collator)
                else:
                    trainer=PerformanceTrainer(model=PerformanceKDModel(student,option),teacher=teacher,options=option,args=args,
                        train_dataset=Rows(96),data_collator=default_data_collator)
                trainer.train()
                trace=trainer.seen if i==0 else trainer.order_trace()
                self.assertTrue(audit_order([trace],range(96),1,option.microbatch)['exact_order'])
                states.append(copy.deepcopy(student.state_dict()))
        for s in states[1:]:
            for k in s:torch.testing.assert_close(s[k],states[0][k],atol=3e-6,rtol=1e-4)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_nonfinite_aborts_before_optimizer(self):
        teacher=tiny_model().requires_grad_(False);student=copy.deepcopy(teacher).requires_grad_(True)
        class Bad(PerformanceKDModel):
            def forward(self,*a,**kw):
                out=super().forward(*a,**kw);out['loss']=out['loss']*float('nan');return out
        initial=copy.deepcopy(student.state_dict());opt=Options(deferred_checks=True)
        with tempfile.TemporaryDirectory() as d:
            trainer=PerformanceTrainer(model=Bad(student,opt),teacher=teacher,options=opt,
                args=make_arguments(d,12,1e-4,1,1,cuda=False,global_sequences=8),
                train_dataset=Rows(96),data_collator=default_data_collator)
            with self.assertRaises(FloatingPointError):trainer.train()
        for k,v in student.state_dict().items():torch.testing.assert_close(v,initial[k],rtol=0,atol=0)

    def test_bounded_options(self):
        for kw in ({'chunk':4096},{'microbatch':8}):
            with self.assertRaises(ValueError):Options(**kw)
        self.assertEqual(VARIANTS['reference'],VARIANTS['reference_repeat'])


class ProbeTests(unittest.TestCase):
    def test_full_gradient_comparison_detects_changes(self):
        from efficient_llm.a2.perf_probe import compare_tensors
        a={'a':torch.tensor([1.,2.]),'b':torch.tensor([3.])}
        same=compare_tensors(a,a)
        self.assertEqual(same['relative_l2'],0.)
        self.assertAlmostEqual(same['cosine'],1.)
        b={k:v*2 for k,v in a.items()}
        self.assertAlmostEqual(compare_tensors(b,a)['relative_l2'],1.)
        with self.assertRaises(ValueError):compare_tensors({'b':a['b']},a)

    def test_launcher_is_bounded_and_uses_same_reference(self):
        from scripts.launch_a2_performance import jobs_for
        from run_experiments import load_jobs
        import json
        jobs=jobs_for(Path('/inputs'),Path('/config'),29500)
        self.assertEqual(len(jobs['jobs']),10)
        self.assertEqual(sum(j['timeout_seconds'] for j in jobs['jobs']),4950)
        for j in jobs['jobs']:
            self.assertEqual(j['gpus'],list(range(8)))
            if j['name']!='reference':self.assertIn('{run_dir}/reference',j['argv'])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'jobs.json';p.write_text(json.dumps(jobs));load_jobs(p)
