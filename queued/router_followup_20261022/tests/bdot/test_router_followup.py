"""Router-only gradient isolation, cosine geometry and six-arm contracts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from test_model import model, batch
from test_grouped import enable_grouped, reference_mm
from bdot import grouped
from bdot.model import BDOTForCausalLM
from bdot.routing import RoutingConfig


def make(fraction=.25, **kwargs):
    return model(route_scope='mlp', routed_mlp_fraction=fraction, curriculum='A',
                 default_prior=0., **kwargs)


class RouterFollowupTests(unittest.TestCase):
    def test_detached_input_blocks_only_input_gradient(self):
        for metric in ('linear','cosine'):
            a=make(router_metric=metric).routed_steps[1].router
            b=copy.deepcopy(a);b.detach_input=True
            h=torch.randn(7,16,requires_grad=True);x=h.detach().clone().requires_grad_()
            visits=torch.randn(7,4);allowed=torch.tensor([[True,True,True,False]]*7)
            za=a(h,visits,allowed);zb=b(x,visits,allowed)
            torch.testing.assert_close(za,zb,atol=0,rtol=0)
            signal=torch.randn(7,3)
            (za[:,:3]*signal).sum().backward();(zb[:,:3]*signal).sum().backward()
            self.assertGreater(float(h.grad.abs().sum()),0);self.assertIsNone(x.grad)
            for p,q in zip(a.parameters(),b.parameters()):
                torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)
            self.assertGreater(float(b.weight.grad.abs().sum()),0)
            self.assertGreater(float(b.norm.weight.grad.abs().sum()),0)

    def test_cosine_formula_and_scale_invariance(self):
        r=make(router_metric='cosine').routed_steps[1].router
        h=torch.randn(9,16,requires_grad=True);v=torch.randn(9,4);allowed=torch.ones(9,4,dtype=torch.bool)
        x=h*torch.rsqrt(h.square().mean(-1,keepdim=True)+r.norm.variance_epsilon)*r.norm.weight
        raw=r.cosine_scale*F.linear(F.normalize(x,dim=-1,eps=1e-6),F.normalize(r.weight,dim=-1,eps=1e-6))+r.bias+F.linear(v,r.visits)
        z=r(h,v,allowed)
        torch.testing.assert_close(z,raw,atol=0,rtol=0)
        scaled=copy.deepcopy(r)
        with torch.no_grad():scaled.weight.mul_(torch.arange(1,5)[:,None]*2.)
        torch.testing.assert_close(z,scaled(h*3.,v,allowed),atol=1e-6,rtol=1e-5)
        z.square().sum().backward();self.assertTrue(torch.isfinite(h.grad).all())
        with torch.no_grad():r.weight.zero_()
        zero=r(torch.zeros_like(h),v,allowed);self.assertTrue(torch.isfinite(zero).all())

    def test_initialization_and_forward_unchanged_by_detach(self):
        for fraction in (.25,.5):
            for metric in ('linear','cosine'):
                a=make(fraction,router_metric=metric,z_loss_coef=0.)
                b=make(fraction,router_metric=metric,router_detach_input=True,z_loss_coef=0.)
                for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,atol=0,rtol=0)
                a.progress=b.progress=.15;a.capture_routes=b.capture_routes=True
                x=a(**batch());y=b(**batch());torch.testing.assert_close(x.logits,y.logits,atol=0,rtol=0)
                self.assertTrue(torch.equal(a.last_routes,b.last_routes))
                y.loss.backward()
                self.assertGreater(float(b.model.embed_tokens.weight.grad.abs().sum()),0)
                for step in b.routed_steps:
                    self.assertGreater(float(step.router.weight.grad.abs().sum()),0)
                    self.assertGreater(float(step.core_mlp.gate_proj.weight.grad.abs().sum()),0)
                self.assertTrue(any(p.grad is not None for e in b.model.bdot_experts for p in e.mlp.parameters()))

    def test_all_variants_replay_grouped_reload_and_causality(self):
        for fraction in (.25,.5):
            for metric,detach in [('linear',True),('cosine',False),('cosine',True)]:
                a=make(fraction,router_metric=metric,router_detach_input=detach)
                b=enable_grouped(copy.deepcopy(a));a.progress=b.progress=.15
                b.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
                a.capture_routes=b.capture_routes=True
                with patch.object(grouped,'_grouped_mm',side_effect=reference_mm):
                    x=a(**batch());y=b(**batch());x.loss.backward();y.loss.backward()
                torch.testing.assert_close(x.logits,y.logits,atol=1e-6,rtol=1e-5)
                self.assertTrue(torch.equal(a.last_routes,b.last_routes))
                for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
                    if p.grad is None:self.assertIsNone(q.grad,name)
                    else:torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-4,msg=name)
                torch.optim.AdamW(a.parameters(),lr=.001).step();a.finish_update()
                with tempfile.TemporaryDirectory() as tmp:
                    a.save_pretrained(tmp);c=BDOTForCausalLM.from_pretrained(tmp,attn_implementation='sdpa')
                for k,v in a.state_dict().items():torch.testing.assert_close(v,c.state_dict()[k],atol=0,rtol=0)
                a.eval();c.eval();out=a(**batch()).logits
                torch.testing.assert_close(out,c(**batch()).logits,atol=0,rtol=0)
                changed=batch();changed['input_ids'][0,:3]=19
                torch.testing.assert_close(out[0,3:],a(**changed).logits[0,3:],atol=1e-6,rtol=1e-5)
                changed=batch();changed['input_ids'][0,2:]=18
                torch.testing.assert_close(out[0,:2],a(**changed).logits[0,:2],atol=1e-6,rtol=1e-5)

    def test_six_configs_change_only_router_and_preserve_parameters(self):
        from scripts.make_bdot_pilot import ROUTER_RUNS
        root=Path(__file__).resolve().parents[2]
        self.assertEqual(len(ROUTER_RUNS),6)
        for name,_ in ROUTER_RUNS:
            base='-'.join(name.split('-')[:2]);c=json.loads((root/'configs/bdot'/(name+'.json')).read_text())
            RoutingConfig(**c['bdot'])
            c['bdot'].pop('router_metric');c['bdot'].pop('router_detach_input')
            self.assertEqual(c,json.loads((root/'configs/bdot'/(base+'.json')).read_text()))
        for kwargs in [dict(router_detach_input=1),dict(router_metric='bad'),
                       dict(router_metric='cosine',init_std_scale=0),dict(router_detach_input=True,routing='random')]:
            with self.assertRaises(ValueError):RoutingConfig(route_scope='mlp',**kwargs)
        with self.assertRaises(ValueError):RoutingConfig(router_detach_input=True)

    def test_followup_queue_and_wait_gate(self):
        from scripts.bdot_router_followup import predecessor_phase
        from scripts.bdot_mlp_pipeline import prepare
        from scripts.make_bdot_pilot import ROUTER_RUNS
        import subprocess,sys
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);old=tmp/'old';(old/'supervised/run').mkdir(parents=True)
            self.assertEqual(predecessor_phase(old),'waiting_for_supervisor')
            state=old/'supervised/supervisor.json'
            state.write_text(json.dumps(dict(training_status='running')))
            self.assertEqual(predecessor_phase(old),'waiting_for_completion')
            state.write_text(json.dumps(dict(training_status='failed')))
            with self.assertRaises(RuntimeError):predecessor_phase(old)
            state.write_text(json.dumps(dict(training_status='ok',finished_at='now')))
            (old/'supervised/run/run.json').write_text(json.dumps(dict(status='ok')))
            (old/'supervised/run/mlp-results.json').write_text(json.dumps(dict(status='ok',runs=[{}]*6)))
            self.assertEqual(predecessor_phase(old),'waiting_for_burn_proof')
            (old/'supervised/burn-verified.json').write_text('{}')
            self.assertEqual(predecessor_phase(old),'verify_burn_handoff') # Still needs live validation.
            state.write_text(json.dumps(dict(training_status='ok',finished_at='now',handoff_error='unknown owner')))
            with self.assertRaises(RuntimeError):predecessor_phase(old)
            pilot=tmp/'pilot'
            subprocess.run([sys.executable,str(root/'scripts/make_bdot_pilot.py'),
                '--output',str(pilot/'prepared'),'--tokenizer',str(tmp/'tok'),'--data',str(tmp/'data'),
                '--eval-data',str(tmp/'eval'),'--revision','a'*40,'--accelerate-config',str(tmp/'accelerate')],
                check=True,capture_output=True)
            jobs=json.loads(prepare(tmp/'new',pilot,study='router').read_text())['jobs']
            self.assertEqual(len(jobs),14)
            self.assertEqual([j['name'] for j in jobs[1:-1:2]],[n+'-s0' for n,_ in ROUTER_RUNS])
            self.assertEqual(jobs[0]['name'],'router_acceptance');self.assertEqual(jobs[-1]['name'],'router_report')
            for job in jobs[:-1]:self.assertEqual(job['gpus'],list(range(8)))

    def test_wait_releases_no_work_before_live_burn_verification(self):
        from types import SimpleNamespace
        from scripts import bdot_router_followup as queue
        from scripts.train_then_burn import BURN_HASH
        for failed in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);old=root/'old';(old/'supervised').mkdir(parents=True)
                (old/'supervised/burn.log').write_text('fixture')
                out=root/'new';out.mkdir();events=[]
                args=SimpleNamespace(host='fixture',output=out,predecessor=old,pilot=root/'pilot',
                    session='test',timeout=30,predecessor_burn_script=root/'burn.py')
                def gate(path):
                    events.append('verified_burn')
                    if failed:raise RuntimeError('unknown worker')
                def inspect():
                    n=events.count('inspect');events.append('inspect')
                    return dict(guard_disabled=n==0)
                with patch.object(queue.socket,'gethostname',return_value='fixture'), \
                     patch.object(queue,'verify_snapshot'),patch.object(queue.shutil,'copytree'), \
                     patch.object(queue,'prepare',return_value=root/'jobs.json'), \
                     patch.object(queue,'predecessor_phase',return_value='verify_burn_handoff'), \
                     patch('scripts.train_then_burn.digest',return_value=BURN_HASH), \
                     patch('scripts.train_then_burn.burn_progress',return_value=True), \
                     patch('scripts.gpu_process.inspect',side_effect=inspect), \
                     patch('scripts.native_baseline_pipeline.burn_gate',side_effect=gate), \
                     patch.object(queue.shutil,'disk_usage',return_value=SimpleNamespace(free=2**50)), \
                     patch.object(queue.time,'sleep',side_effect=lambda _:events.append('wait')), \
                     patch.object(queue.subprocess,'run',side_effect=lambda *a,**kw:events.append('launch')) as launch:
                    if failed:
                        with self.assertRaisesRegex(RuntimeError,'unknown worker'):queue.wait(args)
                        launch.assert_not_called()
                    else:
                        queue.wait(args);launch.assert_called_once()
                        self.assertEqual(events,['inspect','wait','inspect','verified_burn','inspect','launch'])
                        self.assertEqual(json.loads((out/'wait-status.json').read_text())['status'],'launched')
                        self.assertIn('--predecessor-burn-script',launch.call_args.args[0])


if __name__=='__main__':unittest.main()
