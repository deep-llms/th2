"""Independent consumer objective, isolation, backend and Trainer acceptance."""
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

from deep_kv import P7_CONSUMER_ARMS, followup_metadata
from deep_kv.proxy_memory import memory_consumer_loss, MemoryPlan
from tests.test_proxy_screen24 import model, fixture
from tests.test_proxy_heads import batch, objective
from tests.test_fa4_baseline import reference_kernel


def response(block,head,u,prediction,rotary,allowed):
    # Deliberately independent eager reference, not the consumer helper,
    # production memory_attention, entries(), or scaled_dot_product_attention.
    a=block.self_attn;shape=(*u.shape[:-1],-1,a.head_dim)
    q=a.q_norm(a.q_proj(u).view(shape)).transpose(1,2)
    k=a.k_norm(a.k_proj(u).view(shape)).transpose(1,2)
    v=a.v_proj(u).view(shape).transpose(1,2)
    q,k=apply_rotary_pos_emb(q,k,*rotary)
    value=prediction.float();p=(value*torch.rsqrt(value.square().mean(-1,keepdim=True)+1e-6)).to(prediction.dtype)
    kp=head.k_norm(head.k_proj(p).view(*p.shape[:-1],2,-1)).transpose(1,2)
    _,kp=apply_rotary_pos_emb(kp,kp,*rotary)
    vp=head.v_proj(p).view(*p.shape[:-1],2,-1).transpose(1,2)
    keys=torch.cat((k[:,-2:],kp),dim=2).repeat_interleave(2,dim=1)
    values=torch.cat((v[:,-2:],vp),dim=2).repeat_interleave(2,dim=1)
    scores=(q[:,-4:]@keys.transpose(-1,-2))*a.scaling
    weights=scores.masked_fill(~torch.cat((allowed,allowed),dim=-1),-torch.inf).softmax(-1)
    return (weights@values).transpose(1,2)


class ConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_independent_objective_full_gradient_routes_and_metrics(self):
        from deep_kv.proxy_training import summarize_proxy
        for arm in P7_CONSUMER_ARMS:
            actual=model(arm);ref=model('P7-simple-targetgrad' if arm.endswith('targetgrad') else 'P7-simple')
            ctx=batch();mlps={};preds={};sources={};hooks=[]
            for i,block in enumerate(ref.backbone.model.layers,1):
                hooks.append(block.mlp.register_forward_hook(lambda m,a,o,i=i:mlps.__setitem__(i,o)))
                if i in ref.layers:
                    hooks.append(ref.heads[str(i)].w1.register_forward_pre_hook(lambda m,a,i=i:sources.__setitem__(i,a[0])))
                    hooks.append(ref.heads[str(i)].w2.register_forward_hook(lambda m,a,o,i=i:preds.__setitem__(i,o)))
            expected=ref(ctx,compute_auxiliary_losses=False)
            for h in hooks:h.remove()
            out=actual(ctx,collect_target_statistics=True)
            torch.testing.assert_close(out['lm_sum'],expected['lm_sum'],rtol=0,atol=0)
            terms=[];effects=[]
            for i in ref.layers:
                block=copy.deepcopy(ref.backbone.model.layers[i-1]).requires_grad_(False)
                head=copy.deepcopy(ref.heads[str(i)]).requires_grad_(False)
                raw=sum(mlps[j].float() for j in range(i,i+4))
                target=ref.normalize_target(raw,i,target_grad=arm.endswith('targetgrad'))
                rotary=ref.backbone.model.rotary_emb(sources[i],ctx.position_ids)
                teacher=response(block,head,sources[i].detach(),target.detach(),rotary,ctx.allowed()).detach()
                student=response(block,head,sources[i].detach(),preds[i],rotary,ctx.allowed())
                effect=(((student-teacher).square().mean((-1,-2))/(teacher.square().mean((-1,-2))+1e-6))*ctx.valid).sum()
                cosine=((1-F.cosine_similarity(preds[i].float(),target,dim=-1,eps=1e-6))*ctx.valid).sum()
                effects.append(effect.detach());terms.append(effect+(0 if arm.endswith('-only') else cosine))
            aux=torch.stack(terms).mean()
            torch.testing.assert_close(out['aux_sum'],aux,rtol=4e-6,atol=4e-6)
            metrics=summarize_proxy(out['statistics'].numpy(),actual)
            self.assertAlmostEqual(metrics['consumer_effect_loss'],float(torch.stack(effects).mean()/out['aux_count']),places=6)
            out['aux_sum'].backward();aux.backward()
            for (name,p),(_,q) in zip(actual.named_parameters(),ref.named_parameters()):
                self.assertEqual(p.grad is None,q.grad is None,(arm,name))
                if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=5e-6,msg=arm+name)
            self.assertEqual(any(p.grad is not None and bool(p.grad.abs().sum()) for p in actual.backbone.parameters()),arm.endswith('targetgrad'))
            for head in actual.heads.values():
                self.assertIsNotNone(head.w1.weight.grad)
                # The response term is frozen, but live cosine targets can
                # reach projections through the original backbone forward.
                self.assertEqual(head.k_proj.weight.grad is not None,arm.endswith('targetgrad'))
                self.assertEqual(head.v_proj.weight.grad is not None,arm.endswith('targetgrad'))
            actual.zero_grad(set_to_none=True);actual(ctx)['lm_sum'].backward()
            for head in actual.heads.values():
                self.assertIsNone(head.w1.weight.grad)
                self.assertIsNotNone(head.k_proj.weight.grad)

    def test_consumer_identity_causality_isolation_and_frozen_parameters(self):
        from deep_kv.fa4 import document_layout
        for backend in ('sdpa','fa4'):
            with patch('deep_kv.fa4.load_kernel',return_value=(reference_kernel,{'version':'CPU-reference'})):
                m=model(P7_CONSUMER_ARMS[0],backend=backend)
            ctx=batch()
            u=torch.randn(*ctx.input_ids.shape,m.backbone.config.hidden_size,requires_grad=True)
            pred=torch.randn_like(u,requires_grad=True);target=torch.randn_like(u,requires_grad=True)
            rotary=m.backbone.model.rotary_emb(u,ctx.position_ids);plan=MemoryPlan(ctx.segments,sample_queries=False)
            mask=document_layout(ctx) if backend=='fa4' else ctx.allowed()
            args=(m.backbone.model.layers[1],m.heads['2'],u,pred,target,mask,rotary,plan)
            valid=torch.zeros_like(ctx.valid);valid[:,2]=True
            loss=memory_consumer_loss(*args,valid,backend,getattr(m,'fa4_kernel',None)).sum()
            loss.backward()
            self.assertIsNone(u.grad);self.assertIsNone(target.grad)
            self.assertTrue(all(p.grad is None for p in m.parameters()))
            self.assertGreater(float(pred.grad.abs().sum()),0)
            self.assertEqual(float(pred.grad[:,3:].abs().max()),0)
            valid=torch.zeros_like(ctx.valid);valid[:,-1]=True
            pred.grad=None
            memory_consumer_loss(*args,valid,backend,getattr(m,'fa4_kernel',None)).sum().backward()
            forbidden=ctx.segments!=ctx.segments[:,-1:]
            self.assertEqual(float(pred.grad[forbidden].abs().max()),0)
            same=memory_consumer_loss(*args[:4],pred.detach(),*args[5:],ctx.valid,backend,getattr(m,'fa4_kernel',None))
            torch.testing.assert_close(same,torch.zeros_like(same),rtol=0,atol=0)

    def test_checkpoint_and_backend_equivalence(self):
        for arm in P7_CONSUMER_ARMS:
            for bf16 in (False,True):
                models=[model(arm),model(arm,checkpoint=True)]
                with patch('deep_kv.fa4.load_kernel',return_value=(reference_kernel,{'version':'CPU-reference'})):
                    models.append(model(arm,backend='fa4'))
                for m in models:
                    with torch.autocast('cpu',dtype=torch.bfloat16,enabled=bf16):objective(m,m(batch())).backward()
                for j,other in enumerate(models[1:]):
                    for (name,p),(_,q) in zip(models[0].named_parameters(),other.named_parameters()):
                        self.assertEqual(p.grad is None,q.grad is None,name)
                        if p.grad is not None and (not bf16 or j==0):
                            torch.testing.assert_close(p.grad,q.grad,rtol=4e-4 if not bf16 else 0,atol=4e-6 if not bf16 else 0,msg=arm+name)
                    if bf16 and j==1:
                        for group in ('backbone','predictor','projections'):
                            def gradients(m):
                                return torch.cat([p.grad.flatten() for n,p in m.named_parameters() if p.grad is not None and
                                    (n.startswith('backbone.') if group=='backbone' else n.startswith('heads.') and
                                     (('.w1.' in n or '.w2.' in n)==(group=='predictor')))])
                            a,b=gradients(models[0]),gradients(other)
                            self.assertLess(float((a-b).norm()/a.norm().clamp_min(1e-8)),.03,(arm,group))

    def test_trainer_resume_eval_reload_and_report(self):
        from tests.test_train import invoke
        from safetensors.torch import load_file
        from eval.models import load_checkpoint
        from deep_kv.report import report
        for arm in P7_CONSUMER_ARMS:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);args={**fixture(root),'arm':arm,'output_dir':str(root/arm)}
                invoke(root,{**args,'stop_after':1});invoke(root,args)
                invoke(root,{**args,'output_dir':str(root/'full')})
                a,b=load_file(root/arm/'model.safetensors'),load_file(root/'full/model.safetensors')
                for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0,msg=arm+k)
                loaded,_,_=load_checkpoint(root/arm,device='cpu',attention_backend='sdpa')
                self.assertEqual(loaded.wrapped.arm,arm)
                self.assertEqual(loaded.wrapped.aux_target_grad,arm.endswith('targetgrad'))
                self.assertTrue(all(not h.lm_predictor for h in loaded.wrapped.heads.values()))
                report(root,[arm])

    def test_two_rank_real_trainer(self):
        for arm in P7_CONSUMER_ARMS:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);recipe=fixture(root,world=2)
                recipe.update(arm=arm,ddp_backend='gloo',ddp_find_unused_parameters=False,logging_steps=2,eval_on_start=False)
                path=root/'recipe.json';path.write_text(json.dumps(recipe));log=root/'ddp.log'
                with log.open('w') as out:
                    p=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                        '--module','tests.test_proxy_screen24','--ddp-worker',str(path)],
                        cwd=Path(__file__).resolve().parents[1],env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1'},
                        stdout=out,stderr=subprocess.STDOUT,timeout=180)
                self.assertEqual(p.returncode,0,log.read_text()[-15000:])
                self.assertEqual(json.loads((Path(recipe['output_dir'])/'result.json').read_text())['global_step'],3)

    def test_audit_counts_frozen_queries_and_live_student_kv(self):
        from types import SimpleNamespace
        from deep_kv.fa4 import trainer_audit
        for arm in P7_CONSUMER_ARMS:
            with tempfile.TemporaryDirectory() as tmp, patch('deep_kv.fa4.load_kernel',return_value=(reference_kernel,{'version':'CPU-reference'})):
                m=model(arm,backend='fa4')
                trainer=SimpleNamespace(model=m,state=SimpleNamespace(global_step=250),
                    args=SimpleNamespace(process_index=0,world_size=1,output_dir=tmp))
                with trainer_audit(trainer,'train'):
                    objective(m,m(batch())).backward()
                r=json.loads(next(Path(tmp).glob('fa4-*.json')).read_text())
                n=len(m.layers);depth=m.backbone.config.num_hidden_layers
                self.assertEqual(r['calls'],depth+3*n)
                self.assertEqual(r['query_gradient_calls'],depth+n)
                self.assertEqual(r['key_gradient_calls'],depth+2*n)
                self.assertEqual(r['value_gradient_calls'],depth+2*n)


if __name__=='__main__':unittest.main()
