"""P7 tuning: equations, routes, isolation, actual Trainer/DDP and resume."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from deep_kv.p7_variants import (TUNING_ARMS, TUNING_SPECS, TARGETGRAD_VARIANTS,
    LM_VARIANTS, NOAUX_ARMS, KEYONLY_ARMS, VALUEONLY_ARMS, settings_overrides)
from deep_kv.proxy import ProxyModel, ProxySettings, compute_budget
from deep_kv.proxy_memory import MemoryPlan
from tests.test_proxy_heads import batch, objective
from tests.test_proxy_screen24 import config as parent_config, fixture as parent_fixture
from tests.test_fa4_baseline import reference_kernel


def config():
    c=parent_config();c.num_attention_heads=16;c.num_key_value_heads=8
    return c


def model(arm,backend='sdpa',checkpoint=False,**kw):
    options={k.removeprefix('proxy_'):v for k,v in settings_overrides(arm).items()}
    return ProxyModel.from_scratch(config(),arm,consumer=2,deep_target=12,
        proxy_settings=ProxySettings(width=8,features=9,chunk_size=2,**{**options,**kw}),
        checkpoint_layers=checkpoint,checkpoint_aux=checkpoint,checkpoint_lm=checkpoint,
        attention_backend=backend,lm_chunk=3)


def activate(m):
    with torch.no_grad():
        for head in m.heads.values():
            for name in ('key_transform','value_transform'):
                if hasattr(head,name):getattr(head,name)[2].weight.normal_(0,.02)
            if hasattr(head,'value_gate'):head.value_gate.add_(torch.linspace(-.4,.4,len(head.value_gate)))


class TuningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_contracts_initialization_budget_and_routes(self):
        parent=model('P7-simple-targetgrad')
        for arm,spec in TUNING_SPECS.items():
            with self.subTest(arm=arm):
                m=model(arm)
                self.assertEqual(m.layers,(2,6) if spec.get('sparse') else (2,4,6,8))
                self.assertEqual(m.settings.lookahead,spec.get('lookahead',4))
                self.assertEqual(m.settings.groups,spec.get('groups',2))
                self.assertEqual(compute_budget(config(),m.settings,8)[arm]['extra_parameters'],
                                 sum(p.numel() for p in m.heads.parameters()))
                for name,p in m.named_parameters():
                    if name.startswith('backbone.') or (name.startswith('heads.') and ('.w1.' in name or '.w2.' in name)):
                        torch.testing.assert_close(p,parent.state_dict()[name],rtol=0,atol=0)
                    if spec.get('groups',2)==2 and name in parent.state_dict():
                        torch.testing.assert_close(p,parent.state_dict()[name],rtol=0,atol=0)
                activate(m)
                sources=[]
                hooks=[h.w1.register_forward_pre_hook(lambda mod,args:sources.append(args[0].requires_grad)) for h in m.heads.values()]
                out=m(batch());out['lm_sum'].backward()
                self.assertTrue(sources);self.assertFalse(any(sources))
                for h in hooks:h.remove()
                for head in m.heads.values():
                    self.assertEqual(head.w1.weight.grad is not None,arm in LM_VARIANTS)
                    for name,p in head.named_parameters():
                        if name.startswith(('k_proj','v_proj','key_transform','value_transform','value_gate')):
                            self.assertIsNotNone(p.grad,name);self.assertGreater(float(p.grad.abs().sum()),0,name)
                m.zero_grad(set_to_none=True);out=m(batch())
                if arm in NOAUX_ARMS:
                    self.assertEqual(float(out['aux_sum']),0);self.assertFalse(out['aux_sum'].requires_grad)
                    with self.assertRaises(ValueError):model(arm,lambda_max=.1)
                else:
                    out['aux_sum'].backward()
                    live=sum(float(p.grad.abs().sum()) for p in m.backbone.parameters() if p.grad is not None)
                    self.assertEqual(live>0,arm in TARGETGRAD_VARIANTS)
                    for head in m.heads.values():
                        self.assertGreater(float(head.w1.weight.grad.abs().sum()),0)
                        for name,p in head.named_parameters():
                            if not name.startswith(('w1','w2')):
                                # A live target also reaches entry projections via
                                # the backbone that produced that target.
                                self.assertEqual(p.grad is not None,arm in TARGETGRAD_VARIANTS,name)
        for arm in TUNING_ARMS:
            options=settings_overrides(arm)
            with self.assertRaises(ValueError):model(arm,groups=4 if options['proxy_groups']!=4 else 2)

    def test_targets_and_huber_independent_oracle(self):
        for arm in TARGETGRAD_VARIANTS:
            m=model(arm);ctx=batch();mlps={}
            handles=[layer.mlp.register_forward_hook(lambda module,args,out,i=i:mlps.setdefault(i,out))
                     for i,layer in enumerate(m.backbone.model.layers,1)]
            out=m(ctx,collect_target_statistics=True,statistics_mode='mean')
            for h in handles:h.remove()
            weights=TUNING_SPECS[arm].get('weights',[1.]*m.settings.lookahead)
            for site in m.layers:
                raw=sum(mlps[site+j].float()*w for j,w in enumerate(weights))
                torch.testing.assert_close(out['center_sums'][m.mean_index[site]],raw.detach().sum((0,1)))
            if arm.endswith('-huber'):
                from deep_kv.proxy_estimators import smooth_l1_loss
                p=torch.randn(2,5,7,requires_grad=True);t=torch.randn_like(p,requires_grad=True)
                valid=torch.tensor([[1,1,0,1,0],[1,1,1,1,0]],dtype=torch.bool)
                cos,loss=smooth_l1_loss(p,t,valid,target_grad=True)
                d=(p-t).abs();expected=(torch.where(d<1,.5*d.square(),d-.5).mean(-1)*valid).sum(-1)
                torch.testing.assert_close(loss[:,0],expected)
                ga=torch.autograd.grad(loss.sum(),(p,t),retain_graph=True)
                gb=torch.autograd.grad(expected.sum(),(p,t))
                for a,b in zip(ga,gb):torch.testing.assert_close(a,b)
                cosine=(F.cosine_similarity(p,t,dim=-1,eps=1e-6)*valid).sum(-1)
                torch.testing.assert_close(cos[:,0],cosine)

    def test_initial_identity_and_attention_call_budget(self):
        parent=model('P7-simple-targetgrad');nonlinear=model('P7-simple-targetgrad-nonlinear')
        with torch.no_grad():
            torch.testing.assert_close(parent.hidden_states(batch())[0],nonlinear.hidden_states(batch())[0],rtol=0,atol=0)
        for arm in TUNING_ARMS:
            with patch('deep_kv.fa4.load_kernel',return_value=(reference_kernel,{})):
                m=model(arm,backend='fa4')
            calls=[];m._fa4_observer=lambda *args:calls.append(args)
            with torch.no_grad():m.hidden_states(batch())
            self.assertEqual(len(calls),12 if arm in VALUEONLY_ARMS else 12+len(m.layers),arm)
            if arm in VALUEONLY_ARMS:
                self.assertTrue(all(args[0].shape[-2]==8 for args in calls))
            if arm.endswith('-gated'):
                # Normal training logs expose all four distinct query-head gates.
                from types import SimpleNamespace
                from deep_kv.proxy_training import ProxyCallback
                callback=ProxyCallback(2500,64)
                callback.trainer=SimpleNamespace(model=m,normalization_metrics={})
                callback.latest={'lm_loss':1.};callback.seconds_per_update=1.
                logs={'loss':1.};callback.enrich_log(logs,SimpleNamespace(global_step=1))
                gate_values=[v for k,v in logs.items() if 'value_mix_head_' in k]
                self.assertEqual(len(gate_values),4*len(m.layers))
                self.assertTrue(all(abs(v-.05)<1e-7 for v in gate_values))

    def test_attention_equations_outputs_and_all_vjps(self):
        for arm in TUNING_ARMS:
            with self.subTest(arm=arm):
                m=model(arm);activate(m);ctx=batch();head=m.heads['2'];block=m.backbone.model.layers[1];a=block.self_attn
                u=torch.randn(1,8,32,requires_grad=True);pred=torch.randn_like(u,requires_grad=True)
                rotary=m.backbone.model.rotary_emb(u,ctx.position_ids);shape=(*u.shape[:-1],-1,a.head_dim)
                q=a.q_norm(a.q_proj(u).view(shape)).transpose(1,2)
                k=a.k_norm(a.k_proj(u).view(shape)).transpose(1,2)
                v=a.v_proj(u).view(shape).transpose(1,2);q,k=apply_rotary_pos_emb(q,k,*rotary)
                g=m.settings.groups;split=q.shape[1]-2*g
                p=pred if arm in LM_VARIANTS else pred.detach()
                p=p/torch.sqrt(p.square().mean(-1,keepdim=True)+1e-6)
                pk=p if not hasattr(head,'key_transform') else p+head.key_transform(p)
                pv=p if not hasattr(head,'value_transform') else p+head.value_transform(p)
                if arm in VALUEONLY_ARMS:kp=k[:,-g:]
                else:
                    kp=head.k_norm(F.linear(pk,head.k_proj.weight).view(*p.shape[:-1],g,-1)).transpose(1,2)
                    _,kp=apply_rotary_pos_emb(kp,kp,*rotary)
                vp=v[:,-g:] if arm in KEYONLY_ARMS else F.linear(pv,head.v_proj.weight).view(*p.shape[:-1],g,-1).transpose(1,2)
                allowed=ctx.allowed()
                def attend(qq,kk,vv,mask):
                    kk=kk.repeat_interleave(qq.shape[1]//kk.shape[1],dim=1)
                    vv=vv.repeat_interleave(qq.shape[1]//vv.shape[1],dim=1)
                    return (((qq@kk.transpose(-1,-2))*a.scaling).masked_fill(~mask,-torch.inf).softmax(-1)@vv).transpose(1,2)
                native=attend(q[:,:split],k[:,:-g],v[:,:-g],allowed)
                if hasattr(head,'value_gate'):
                    # Independent oracle: blend two attention OUTPUTS rather than values.
                    x=attend(q[:,split:],k[:,-g:],v[:,-g:],allowed)
                    y=attend(q[:,split:],kp,vp,allowed)
                    weight=head.value_gate.sigmoid()[None,None,:,None]
                    proxy=(1-weight)*x+weight*y
                else:
                    proxy=attend(q[:,split:],torch.cat((k[:,-g:],kp),dim=2),
                        torch.cat((v[:,-g:],vp),dim=2),torch.cat((allowed,allowed),dim=-1))
                expected=torch.cat((native,proxy),dim=2)
                actual=m.memory_attention(block,u,head,pred,allowed,rotary,MemoryPlan(ctx.segments,sample_queries=False))
                torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
                params=(u,pred,*head.parameters(),*a.parameters())
                ga=torch.autograd.grad(actual.square().sum(),params,allow_unused=True,retain_graph=True)
                gb=torch.autograd.grad(expected.square().sum(),params,allow_unused=True)
                for x,y in zip(ga,gb):
                    if x is None:self.assertIsNone(y)
                    else:torch.testing.assert_close(x,y,rtol=1e-4,atol=1e-5)

    def test_backends_checkpoint_replay_and_isolation(self):
        for arm in TUNING_ARMS:
            for bf16 in (False,True):
                with self.subTest(arm=arm,bf16=bf16):
                    dense=model(arm);activate(dense);checked=model(arm,checkpoint=True)
                    with patch('deep_kv.fa4.load_kernel',return_value=(reference_kernel,{})):flash=model(arm,backend='fa4')
                    for m in (checked,flash):m.load_state_dict(dense.state_dict())
                    outs=[]
                    for m in (dense,checked,flash):
                        with torch.autocast('cpu',dtype=torch.bfloat16,enabled=bf16):
                            out=m(batch(),collect_target_statistics=True);objective(m,out).backward();outs.append(out)
                    for other,out in zip((checked,flash),outs[1:]):
                        for group in ('backbone.','heads.'):
                            def grads(m):return torch.cat([p.grad.flatten().float() for n,p in m.named_parameters() if n.startswith(group) and p.grad is not None])
                            a,b=grads(dense),grads(other)
                            if other is checked:torch.testing.assert_close(a,b,rtol=0,atol=0)
                            else:self.assertLess(float((a-b).norm()/a.norm().clamp_min(1e-8)),.03 if bf16 else 1e-4)
                        torch.testing.assert_close(out['lm_sum'],outs[0]['lm_sum'],rtol=.002 if bf16 else 1e-5,atol=1e-5)
                    ctx=batch();changed=copy.deepcopy(ctx);changed.input_ids[:,:3]=17
                    with torch.no_grad():
                        a=dense.hidden_states(ctx)[0];b=dense.hidden_states(changed)[0]
                    torch.testing.assert_close(a[:,3:],b[:,3:],rtol=0,atol=0)
                    changed=copy.deepcopy(ctx);changed.input_ids[:,6:]=18
                    with torch.no_grad():b=dense.hidden_states(changed)[0]
                    torch.testing.assert_close(a[:,:6],b[:,:6],rtol=0,atol=0)

    def test_gated_sdpa_training_audit_attribution(self):
        from deep_kv.sdpa_audit import SDPAAudit
        m=model('P7-simple-lm-valueonly-gated');audit=SDPAAudit(m)
        with audit.observe():
            out=m(batch());loss=objective(m,out)
            loss.register_hook(audit.backward_marker);loss.backward()
        self.assertEqual(len(audit.calls),12)
        self.assertTrue(all(c['site'].startswith('backbone.layer_') for c in audit.calls))
        self.assertFalse(audit.unattributed_backward)

    def test_real_trainer_resume_reload_and_report(self):
        from tests.test_train import invoke
        from safetensors.torch import load_file
        from eval.models import load_checkpoint
        from deep_kv.report import report
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);recipe=parent_fixture(root);config().save_pretrained(root/'model')
            for arm in TUNING_ARMS:
                args={**recipe,**settings_overrides(arm),'arm':arm,'output_dir':str(root/arm)}
                invoke(root,{**args,'stop_after':1});invoke(root,args)
                invoke(root,{**args,'output_dir':str(root/('full-'+arm))})
                a=load_file(root/arm/'model.safetensors');b=load_file(root/('full-'+arm)/'model.safetensors')
                for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0,msg=arm+'/'+key)
                loaded,_,_=load_checkpoint(root/arm,device='cpu',attention_backend='sdpa')
                self.assertEqual(loaded.wrapped.arm,arm);report(root,[arm])
                cfg=json.loads((root/arm/'train_config.json').read_text())
                bad=copy.deepcopy(cfg);bad['proxy_target']['memory_entries']['kv_groups']=99
                (root/arm/'train_config.json').write_text(json.dumps(bad))
                with self.assertRaises(ValueError):report(root,[arm])
                (root/arm/'train_config.json').write_text(json.dumps(cfg))
            report(root,list(TUNING_ARMS))

    def test_two_rank_trainer_all_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);recipe=parent_fixture(root,world=2);config().save_pretrained(root/'model')
            for arm in TUNING_ARMS:
                r={**recipe,**settings_overrides(arm),'arm':arm,'output_dir':str(root/arm),
                    'ddp_backend':'gloo','ddp_find_unused_parameters':False,'logging_steps':2,'eval_on_start':False}
                path=root/'ddp.json';path.write_text(json.dumps(r));log=root/'ddp.log'
                with log.open('w') as output:
                    p=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                        '--module','tests.test_proxy_screen24','--ddp-worker',str(path)],
                        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1'},
                        stdout=output,stderr=subprocess.STDOUT,timeout=180)
                self.assertEqual(p.returncode,0,arm+'\n'+log.read_text()[-12000:])
                self.assertEqual(json.loads((root/arm/'result.json').read_text())['global_step'],3)

    def test_queue_is_explicit_and_smokes_precede_training(self):
        from scripts.p7_tuning_queue import CASES, make
        self.assertEqual(len(CASES),20)
        self.assertEqual(set(TUNING_ARMS),{a for _,a,_,_ in CASES if a in TUNING_ARMS})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);jobs=make('proxy_heads.12layer.b200.json',root/'jobs.json','/local/rows.json')
            self.assertEqual(len(jobs),81)
            for label,arm,weight,seed in CASES:
                job=next(j for j in jobs if j['name']==label+'-train');argv=job['argv']
                get=lambda k:argv[argv.index('--'+k)+1]
                self.assertEqual(get('seed'),str(seed));self.assertEqual(get('data_seed'),str(seed))
                self.assertEqual(get('max_steps'),'2500');self.assertEqual(get('warmup_steps'),'125')
                self.assertEqual(get('save_total_limit'),'0');self.assertEqual(get('save_steps'),'250')
                self.assertEqual(float(get('proxy_lambda_max')),weight);self.assertEqual(job['gpus'],list(range(8)))
                self.assertGreater(jobs.index(job),max(i for i,j in enumerate(jobs) if j['name'].startswith('smoke-')))
            with self.assertRaises(FileExistsError):make('proxy_heads.12layer.b200.json',root/'jobs.json','rows')


if __name__=='__main__':unittest.main()
