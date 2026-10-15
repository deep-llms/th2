"""P7 LM ablations: independent attention/CE, routes, Trainer resume and DDP."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch
from torch.nn import functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from deep_kv import P7_EXPLORATION_ARMS, followup_metadata
from deep_kv.proxy import compute_budget
from deep_kv.proxy_memory import MemoryPlan, next_token_loss, next_token_mask
from deep_kv.proxy_training import ProxyTrainer, summarize_proxy
from tests.test_proxy_screen24 import model, config, fixture
from tests.test_proxy_heads import batch, objective
from tests.test_fa4_baseline import reference_kernel


class ExplorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_initialization_layout_budget_and_routes(self):
        parent = model('P7-simple-lm')
        for arm in P7_EXPLORATION_ARMS:
            m = model(arm)
            for name, value in m.state_dict().items():
                if name in ('mu','sigma2') and arm.endswith('-sparse'):continue
                torch.testing.assert_close(value,parent.state_dict()[name],rtol=0,atol=0,msg=arm+'/'+name)
            self.assertEqual(m.layers,(2,6) if arm.endswith('-sparse') else (2,4,6,8))
            self.assertEqual(compute_budget(config(),m.settings,8)[arm]['extra_parameters'],
                             sum(p.numel() for p in m.heads.parameters()))
            if arm.endswith('-valueonly'):
                budget=compute_budget(config(),m.settings,8)[arm]
                self.assertEqual(budget['extra_macs_per_token'],budget['fa4_extra_macs_per_token'])
                self.assertEqual(budget['extra_macs_per_token'],budget['extra_parameters'])
            self.assertTrue(followup_metadata(arm)['lm_to_predictor'])
            sources=[]
            hooks=[h.w1.register_forward_pre_hook(lambda module,args:sources.append(args[0].requires_grad)) for h in m.heads.values()]
            out=m(batch());out['lm_sum'].backward()
            self.assertTrue(sources);self.assertFalse(any(sources))
            for hook in hooks:hook.remove()
            self.assertTrue(all(h.w1.weight.grad is not None and h.w1.weight.grad.abs().sum()>0 for h in m.heads.values()))
            m.zero_grad(set_to_none=True);out=m(batch())
            if arm.endswith('-noaux'):
                self.assertEqual(float(out['aux_sum']),0);self.assertEqual(m.auxiliary_weight(2500),0)
                self.assertEqual(summarize_proxy(out['statistics'].numpy(),m)['aux_loss'],0)
                with self.assertRaises(ValueError):model(arm,lambda_max=.1)
                continue
            out['aux_sum'].backward()
            self.assertTrue(all(p.grad is None for p in m.backbone.parameters()))
            for key,head in m.heads.items():
                live = not arm.endswith('-nexttoken') or int(key)==m.layers[-1]
                self.assertEqual(head.w1.weight.grad is not None,live)
                for projection in (head.k_proj,head.v_proj):
                    if projection is not None:self.assertIsNone(projection.weight.grad)

    def test_key_value_variants_against_independent_joint_attention(self):
        for arm in ('P7-simple-lm-keyonly','P7-simple-lm-valueonly'):
            m=model(arm);ctx=batch();head=m.heads['2'];block=m.backbone.model.layers[1];a=block.self_attn
            u=torch.randn(1,8,32,requires_grad=True);pred=torch.randn_like(u,requires_grad=True)
            rotary=m.backbone.model.rotary_emb(u,ctx.position_ids);shape=(*u.shape[:-1],-1,a.head_dim)
            q=a.q_norm(a.q_proj(u).view(shape)).transpose(1,2)
            k=a.k_norm(a.k_proj(u).view(shape)).transpose(1,2)
            v=a.v_proj(u).view(shape).transpose(1,2);q,k=apply_rotary_pos_emb(q,k,*rotary)
            p=pred*torch.rsqrt(pred.square().mean(-1,keepdim=True)+1e-6)
            if arm.endswith('-keyonly'):
                kp=head.k_norm(head.k_proj(p).view(*p.shape[:-1],2,-1)).transpose(1,2)
                _,kp=apply_rotary_pos_emb(kp,kp,*rotary);vp=v[:,-2:]
            else:kp=k[:,-2:];vp=head.v_proj(p).view(*p.shape[:-1],2,-1).transpose(1,2)
            keys=torch.cat((k[:,-2:],kp),dim=2).repeat_interleave(2,dim=1)
            vals=torch.cat((v[:,-2:],vp),dim=2).repeat_interleave(2,dim=1)
            scores=(q[:,-4:]@keys.transpose(-1,-2))*a.scaling
            allowed=torch.cat((ctx.allowed(),ctx.allowed()),dim=-1)
            expected=(scores.masked_fill(~allowed,-torch.inf).softmax(-1)@vals).transpose(1,2)
            actual=m.memory_attention(block,u,head,pred,ctx.allowed(),rotary,MemoryPlan(ctx.segments))[:,:,-4:]
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
            params=(u,pred,*head.parameters(),*a.parameters())
            ga=torch.autograd.grad(actual.square().sum(),params,allow_unused=True,retain_graph=True)
            gb=torch.autograd.grad(expected.square().sum(),params,allow_unused=True)
            for x,y in zip(ga,gb):
                if x is None:self.assertIsNone(y)
                else:torch.testing.assert_close(x,y,rtol=4e-5,atol=3e-6)

    def test_next_token_independent_ce_mask_and_frozen_readout(self):
        # Multiple selected positions, including boundary and ignored labels.
        from deep_kv.model import Context
        ids=torch.arange(24).view(1,-1)%31;segments=torch.tensor([[0]*9+[1]*15])
        labels=ids.clone();labels[:,17]=-100
        ctx=Context(ids,torch.ones_like(ids,dtype=torch.bool),torch.arange(24).view(1,-1),segments,labels)
        p=torch.randn(1,24,32,requires_grad=True);norm=torch.randn(32,requires_grad=True)
        weight=torch.randn(32,32,requires_grad=True)
        actual=next_token_loss(p,ctx,norm,weight,1e-6,2)
        selected=[t for t in range(0,23,8) if segments[0,t]==segments[0,t+1] and labels[0,t+1]!=-100]
        self.assertEqual(selected,[0]);self.assertEqual(int(next_token_mask(ctx).sum()),1)
        h=p[:,selected].float();h=h/torch.sqrt(h.square().mean(-1,keepdim=True)+1e-6)
        logits=(h*norm.detach())@weight.detach().T
        expected=F.cross_entropy(logits.flatten(0,1),labels[:,[t+1 for t in selected]].flatten(),reduction='sum')
        torch.testing.assert_close(actual.sum(),expected)
        gp=torch.autograd.grad(expected,p,retain_graph=True)[0];actual.sum().backward()
        torch.testing.assert_close(p.grad,gp);self.assertIsNone(norm.grad);self.assertIsNone(weight.grad)
        self.assertEqual(float(p.grad[:,1:].abs().sum()),0)
        masked=copy.deepcopy(ctx);masked.labels.fill_(-100)
        zero=next_token_loss(p,masked,norm,weight,1e-6,2)
        self.assertEqual(float(zero.sum()),0)
        self.assertEqual(float(torch.autograd.grad(zero.sum(),p)[0].abs().sum()),0)

    def test_value_merge_full_model_and_attention_call_count(self):
        from deep_kv.proxy import ProxyModel
        def duplicate_entries(self,a,q,k,v,vp,mask,plan):
            return self.joint_memory_attention(a,q,k,v,k[:,-2:],vp,mask,plan)
        for backend in ('sdpa','fa4'):
            for bf16 in (False,True):
                with self.subTest(backend=backend,bf16=bf16):
                    with patch('deep_kv.fa4.load_kernel',return_value=(reference_kernel,{})):
                        merged=model('P7-simple-lm-valueonly',backend=backend)
                        reference=model('P7-simple-lm-valueonly',backend=backend)
                    calls=[]
                    if backend=='fa4':merged._fa4_observer=lambda *args:calls.append(args)
                    with torch.autocast('cpu',dtype=torch.bfloat16) if bf16 else nullcontext():
                        out=merged(batch(),collect_target_statistics=True);objective(merged,out).backward()
                        with patch.object(ProxyModel,'value_memory_attention',duplicate_entries):
                            old=reference(batch(),collect_target_statistics=True);objective(reference,old).backward()
                    self.assertLess(float((out['lm_sum']/out['lm_count']-old['lm_sum']/old['lm_count']).detach().abs()),.01 if bf16 else 1e-6)
                    self.assertLess(float((out['aux_sum']/out['aux_count']-old['aux_sum']/old['aux_count']).detach().abs()),.01 if bf16 else 1e-6)
                    for prefix in ('heads.','backbone.'):
                        pairs=[(p,q) for (name,p),(_,q) in zip(merged.named_parameters(),reference.named_parameters()) if name.startswith(prefix)]
                        for p,q in pairs:self.assertEqual(p.grad is None,q.grad is None)
                        a=torch.cat([p.grad.flatten().float() for p,q in pairs if p.grad is not None])
                        b=torch.cat([q.grad.flatten().float() for p,q in pairs if q.grad is not None])
                        self.assertLess(float((a-b).norm()/b.norm().clamp_min(1e-8)),.03 if bf16 else 2e-5)
                    merged.zero_grad(set_to_none=True);reference.zero_grad(set_to_none=True)
                    with torch.no_grad(),(torch.autocast('cpu',dtype=torch.bfloat16) if bf16 else nullcontext()):
                        a=merged.hidden_states(batch())[0]
                        with patch.object(ProxyModel,'value_memory_attention',duplicate_entries):b=reference.hidden_states(batch())[0]
                        self.assertLess(float((a.float()-b.float()).norm()/b.float().norm()),.02 if bf16 else 1e-6)
                    if backend=='fa4':
                        self.assertEqual(len(calls),2*config().num_hidden_layers)
                        self.assertTrue(all(args[0].shape[-2]==8 for args in calls))

    def test_backend_checkpoint_equivalence_and_document_isolation(self):
        for arm in P7_EXPLORATION_ARMS:
            dense=model(arm);checked=model(arm,checkpoint=True)
            with patch('deep_kv.fa4.load_kernel',return_value=(reference_kernel,{})):
                flash=model(arm,backend='fa4')
            outputs=[]
            for m in (dense,checked,flash):
                o=m(batch(),collect_target_statistics=True);objective(m,o).backward();outputs.append(o)
            for other,out in zip((checked,flash),outputs[1:]):
                for key in outputs[0]:torch.testing.assert_close(out[key],outputs[0][key],rtol=5e-5,atol=5e-6,msg=arm+'/'+key)
                for (name,p),(_,q) in zip(dense.named_parameters(),other.named_parameters()):
                    self.assertEqual(p.grad is None,q.grad is None,name)
                    if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=5e-4,atol=5e-6,msg=name)
            ctx=batch();changed=copy.deepcopy(ctx);changed.input_ids[:,:3]=17
            with torch.no_grad():
                a=dense.hidden_states(ctx)[0];b=dense.hidden_states(changed)[0]
            torch.testing.assert_close(a[:,3:],b[:,3:],rtol=0,atol=0)

    def test_bf16_routes_and_backend_gradients(self):
        for arm in P7_EXPLORATION_ARMS:
            models=[model(arm),model(arm,checkpoint=True)]
            with patch('deep_kv.fa4.load_kernel',return_value=(reference_kernel,{})):
                models.append(model(arm,backend='fa4'))
            for m in models:
                with torch.autocast('cpu',dtype=torch.bfloat16):out=m(batch());objective(m,out).backward()
                self.assertTrue(torch.isfinite(out['lm_sum']))
                self.assertTrue(torch.isfinite(out['aux_sum']))
            for other in models[1:]:
                for group in ('backbone.','heads.'):
                    def grads(m):
                        return torch.cat([p.grad.flatten().float() for n,p in m.named_parameters()
                                          if n.startswith(group) and p.grad is not None])
                    a,b=grads(models[0]),grads(other)
                    if other is models[1]:torch.testing.assert_close(a,b,rtol=0,atol=0)
                    else:self.assertLess(float((a-b).norm()/a.norm().clamp_min(1e-8)),.03,(arm,group))

    def test_trainer_resume_reload_report_and_accumulation_counts(self):
        from tests.test_train import invoke
        from safetensors.torch import load_file
        from eval.models import load_checkpoint
        from deep_kv.report import report
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);recipe=fixture(root)
            for arm in P7_EXPLORATION_ARMS:
                args={**recipe,'arm':arm,'output_dir':str(root/arm)}
                invoke(root,{**args,'stop_after':1});invoke(root,args)
                invoke(root,{**args,'output_dir':str(root/('full-'+arm))})
                a=load_file(root/arm/'model.safetensors');b=load_file(root/('full-'+arm)/'model.safetensors')
                for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0,msg=arm+'/'+key)
                loaded,_,_=load_checkpoint(root/arm,device='cpu',attention_backend='sdpa')
                self.assertEqual(loaded.wrapped.arm,arm);report(root,[arm])
            report(root,list(P7_EXPLORATION_ARMS))
        # Verify the Trainer denominator over microbatches with unequal eligible
        # counts; this must be sampled target count, not all packed input tokens.
        from types import SimpleNamespace
        ctx=batch();inputs=dict(input_ids=ctx.input_ids,labels=ctx.labels,attention_mask=ctx.valid.long(),segments=ctx.segments)
        empty={**inputs,'labels':torch.full_like(ctx.labels,-100)}
        stub=SimpleNamespace(model=model('P7-simple-lm-nexttoken'),context=ProxyTrainer.context,
            accelerator=SimpleNamespace(reduce=lambda x,**kw:x*2,num_processes=2,process_index=0))
        counts=ProxyTrainer._get_num_items_in_batch(stub,[inputs,empty],torch.device('cpu'))
        self.assertEqual(counts.tolist(),[12,2])

    def test_two_rank_trainer_all_new_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);recipe=fixture(root,world=2)
            for arm in P7_EXPLORATION_ARMS:
                r={**recipe,'arm':arm,'output_dir':str(root/arm),'ddp_backend':'gloo',
                   'ddp_find_unused_parameters':False,'logging_steps':2,'eval_on_start':False}
                path=root/'ddp.json';path.write_text(json.dumps(r));log=root/'ddp.log'
                with log.open('w') as output:
                    p=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                        '--module','tests.test_proxy_screen24','--ddp-worker',str(path)],
                        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1'},
                        stdout=output,stderr=subprocess.STDOUT,timeout=180)
                self.assertEqual(p.returncode,0,log.read_text()[-15000:])
                self.assertEqual(json.loads((root/arm/'result.json').read_text())['global_step'],3)


if __name__=='__main__':unittest.main()
