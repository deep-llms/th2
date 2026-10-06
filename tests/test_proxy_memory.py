"""P7 contract; FA4 CPU tests use an independent per-document attention oracle."""
import copy
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from datasets import Dataset
from transformers import TrainingArguments, AutoTokenizer, TrainerControl

from deep_kv import MEMORY_ARMS
from deep_kv.fa4 import document_layout
from deep_kv.model import Context
from deep_kv.proxy import ProxyModel, ProxySettings, EMSPlan, compute_budget
from deep_kv.proxy_memory import MemoryPlan, rectangular_mask, relational_losses, flash_joint_attention
from deep_kv.proxy_training import ProxyTrainer, ProxyCallback, summarize_proxy
from deep_kv.packing import isolated_data_collator
from deep_kv.report import report
from deep_kv.__main__ import jobs
from tests.test_proxy_heads import cfg, batch
from tests.test_fa4_baseline import reference_kernel
from tests.test_proxy_training import proxy_fixture
from tests.test_train import invoke


def config():
    c=cfg();c.num_attention_heads=16;c.num_key_value_heads=8
    return c


def model(arm='P7',backend='sdpa',checkpoint=False,**settings):
    return ProxyModel.from_scratch(config(),arm,consumer=2,deep_target=8,
        proxy_settings=ProxySettings(width=8,features=9,chunk_size=2,**settings),
        attention_backend=backend,checkpoint_layers=checkpoint,checkpoint_aux=checkpoint,
        checkpoint_lm=checkpoint,lm_chunk=3)


def objective(out):
    return out['lm_sum']/out['lm_count']+.1*(out['aux_sum']/out['aux_count'].clamp_min(1)+
                                           (.5*out['rel_sum']/out['rel_count'].clamp_min(1) if 'rel_sum' in out else 0))


def loop_relational_reference(prediction, target, plan):
    """Original per-sequence implementation, retained only as a numerical oracle."""
    with torch.autocast(prediction.device.type, enabled=False):
        pred=F.normalize(prediction.float(),dim=-1,eps=1e-6)
        truth=F.normalize(target.detach().float(),dim=-1,eps=1e-6)
        sums=[];counts=[];positions=torch.arange(pred.shape[1],device=pred.device)
        for row,queries in enumerate(plan.queries):
            if not len(queries):
                sums.append(pred[row].sum()*0);counts.append(0);continue
            allowed=(positions[None]<queries[:,None]) & (plan.documents[row,queries,None]==plan.documents[row,None,:])
            p=((pred[row,queries]@pred[row].T)/.1).masked_fill(~allowed,-torch.inf).log_softmax(-1)
            t=((truth[row,queries]@truth[row].T)/.1).masked_fill(~allowed,-torch.inf).log_softmax(-1)
            sums.append((t.exp()*(t.masked_fill(~allowed,0)-p.masked_fill(~allowed,0))).sum())
            counts.append(len(queries))
        return torch.stack(sums),pred.new_tensor(counts)


class MemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        kernel=patch('deep_kv.fa4.load_kernel',return_value=(reference_kernel,{'version':'CPU-test-double'}))
        kernel.start();self.addCleanup(kernel.stop)

    def test_t0_t1_t9_placement_initialization_and_masked_equivalence(self):
        c=config();c.num_hidden_layers=28;c.layer_types=['full_attention']*28
        full=ProxyModel.from_scratch(c,'P7',proxy_settings=ProxySettings(width=8,chunk_size=2))
        self.assertEqual(full.layers,tuple(range(2,25,2)))
        for backend in ('sdpa','fa4'):
            base=model('A',backend);expected=base.backbone.lm_head(base.hidden_states(batch())[0])
            for arm in MEMORY_ARMS:
                m=model(arm,backend)
                for name,value in base.backbone.state_dict().items():
                    torch.testing.assert_close(value,m.backbone.state_dict()[name],rtol=0,atol=0)
                self.assertFalse(any('alpha' in name for name,_ in m.heads.named_parameters()))
                with m.without_proxy():actual=m.backbone.lm_head(m.hidden_states(batch())[0])
                torch.testing.assert_close(expected,actual,rtol=0,atol=1e-5)
                other=model(arm,backend,module_seed=77)
                self.assertFalse(torch.equal(next(m.heads['2'].parameters()),next(other.heads['2'].parameters())))
                self.assertTrue(torch.equal(m.backbone.model.embed_tokens.weight,other.backbone.model.embed_tokens.weight))
        for options in (dict(groups=1),dict(alpha_init=1),dict(isolate_estimator=False),dict(lookahead=3),
                        dict(lambda_max=float('nan')),dict(lambda_max=float('inf'))):
            with self.assertRaises(ValueError):model(**options)

    def test_t2_t4_joint_softmax_native_heads_and_exact_gqa(self):
        for arm in ('P7','P7-simple'):
            m=model(arm);ctx=batch();u=torch.randn(1,8,32);head=m.heads['2'];layer=m.backbone.model.layers[1]
            rotary=m.backbone.model.rotary_emb(u,ctx.position_ids);plan=MemoryPlan(ctx.segments)
            prediction=head.estimate(u,plan);calls=[]
            def attention(module,q,k,v,mask,**kwargs):
                calls.append((q,k,v,mask))
                return F.scaled_dot_product_attention(q,k,v,attn_mask=mask,enable_gqa=True,
                                                     scale=module.scaling).transpose(1,2),None
            with patch('deep_kv.proxy.ALL_ATTENTION_FUNCTIONS.get_interface',return_value=attention):
                native=m.attention(layer,u,None,ctx.allowed(),rotary)
                joint=m.memory_attention(layer,u,head,prediction,ctx.allowed(),rotary,plan)
            torch.testing.assert_close(native[:,:,:12],joint[:,:,:12],rtol=0,atol=1e-6)
            for index,heads in ((0,12),(1,6),(2,6)):
                torch.testing.assert_close(calls[0][index][:,:heads],calls[1][index],rtol=0,atol=0)
            q,k,v,mask=calls[2]
            self.assertEqual((q.shape[1],k.shape[1],k.shape[2]),(4,2,16))
            torch.testing.assert_close(mask,torch.cat((ctx.allowed(),ctx.allowed()),dim=-1),rtol=0,atol=0)
            manual=((q @ k.repeat_interleave(2,1).transpose(-1,-2))*layer.self_attn.scaling).masked_fill(~mask,-torch.inf).softmax(-1)
            expected=(manual @ v.repeat_interleave(2,1)).transpose(1,2)
            torch.testing.assert_close(joint[:,:,12:],expected,rtol=1e-5,atol=1e-6)
            ablated=rectangular_mask(ctx.allowed(),True)
            self.assertFalse(ablated[...,8:].any());self.assertTrue(ablated.any(-1).all())
            for t in range(8):
                for j in range(8):
                    self.assertEqual(bool(mask[0,0,t,j+8]),j<=t and ctx.segments[0,j]==ctx.segments[0,t])

    def test_fa4_joint_reference_values_gradients_and_native_value_variant(self):
        ctx=batch();layout=document_layout(ctx)
        torch.manual_seed(91)
        values=[torch.randn(1,h,8,8) for h in (4,2,2,2,2)]
        results=[]
        for flash in (False,True):
            q,k,v,kp,vp=[x.clone().requires_grad_() for x in values]
            if flash:out=flash_joint_attention(q,k,v,kp,vp,layout,.2,reference_kernel)
            else:out=F.scaled_dot_product_attention(q,torch.cat((k,kp),2),torch.cat((v,vp),2),
                attn_mask=rectangular_mask(ctx.allowed()),enable_gqa=True,scale=.2).transpose(1,2)
            out.square().sum().backward();results.append((out,*[x.grad for x in (q,k,v,kp,vp)]))
        for a,b in zip(*results):torch.testing.assert_close(a,b,rtol=1e-5,atol=2e-6)
        m=model('P7-kq');head=m.heads['2'];native=torch.randn(1,2,8,8);u=torch.randn(1,8,32)
        rotary=m.backbone.model.rotary_emb(u,ctx.position_ids)
        _,vp=head.entries(head.estimate(u,MemoryPlan(ctx.segments)),rotary,native)
        self.assertIs(vp,native);self.assertIsNone(head.v_proj)

    def test_t3_t5_causality_documents_targets_and_gradients(self):
        for backend in ('sdpa','fa4'):
            for arm in MEMORY_ARMS:
                m=model(arm,backend);ctx=batch();records=[];targets=[];predictions=[]
                hook=m.backbone.model.embed_tokens.register_forward_hook(lambda module,args,out:records.append(out))
                original=m.normalize_target;estimate=m.heads['2'].estimate
                def normalized(value,*args,**kwargs):
                    targets.append(value.detach().clone());return original(value,*args,**kwargs)
                def estimated(*args):
                    out=estimate(*args);predictions.append(out);return out
                changed=copy.deepcopy(ctx);changed.input_ids[:,:3]=20
                with patch.object(m,'normalize_target',side_effect=normalized),patch.object(m.heads['2'],'estimate',side_effect=estimated):
                    hidden,*_=m._run_backbone(ctx,compute_auxiliary_losses=True,auxiliary_grad=True,collect_target_statistics=False)
                    grad=torch.autograd.grad(hidden[:,3:].square().sum(),records[-1])[0]
                    self.assertEqual(grad[:,:3].count_nonzero(),0)
                    target_before=[x.clone() for x in targets];targets.clear()
                    other,*_=m._run_backbone(changed,compute_auxiliary_losses=True,auxiliary_grad=True,collect_target_statistics=False)
                hook.remove()
                torch.testing.assert_close(hidden[:,3:],other[:,3:],rtol=0,atol=0)
                for a,b in zip(target_before,targets):torch.testing.assert_close(a[:,3:],b[:,3:],rtol=0,atol=0)
                torch.testing.assert_close(predictions[0][:,3:],predictions[1][:,3:],rtol=0,atol=0)
                plan=MemoryPlan(ctx.segments,ems=EMSPlan(ctx.segments,m.gamma,2) if arm=='P7-ems' else None)
                u=torch.randn(1,8,32);future=u.clone();future[:,6:]+=10
                torch.testing.assert_close(estimate(u,plan)[:,:6],estimate(future,plan)[:,:6],rtol=0,atol=0)
        # Relational loss on document two must not read document one or itself.
        plan=MemoryPlan(batch().segments);plan.queries=[torch.tensor([4,5,6,7])]
        pred=torch.randn(1,8,7,requires_grad=True);truth=torch.randn_like(pred)
        rel,_=relational_losses(pred,truth,plan)
        grad=torch.autograd.grad(rel.sum(),pred)[0]
        self.assertEqual(grad[:,:3].count_nonzero(),0)
        manual=[]
        for i in plan.queries[0]:
            p=F.normalize(pred[0].float(),dim=-1,eps=1e-6);t=F.normalize(truth[0],dim=-1,eps=1e-6)
            target=(t[3:i]@t[i]/.1).softmax(-1)
            manual.append((target*(target.log()-(p[3:i]@p[i]/.1).log_softmax(-1))).sum())
        torch.testing.assert_close(rel.sum(),torch.stack(manual).sum(),rtol=1e-5,atol=1e-6)

    def test_t6_t8_routing_and_exact_target_quantities(self):
        for arm in MEMORY_ARMS:
            for backend in ('sdpa','fa4'):
                m=model(arm,backend);ctx=batch()
                for component in (('lm_sum','aux_sum','rel_sum') if m.relational_proxy else ('lm_sum','aux_sum')):
                    m.zero_grad(set_to_none=True);out=m(ctx);out[component].backward()
                    for name,p in m.named_parameters():
                        active=p.grad is not None and bool(p.grad.count_nonzero())
                        estimator=name.startswith('heads.') and any(token in name for token in ('.w_in.','.w_out.','.conv','.w1.','.w2.'))
                        if component=='lm_sum' and estimator:self.assertIsNone(p.grad,name)
                        elif component!='lm_sum':self.assertEqual(active,estimator,(component,name))
                        elif name.startswith('heads.'):self.assertTrue(active,name)
                blocks={};raw={};mlps={};hooks=[]
                for i,layer in enumerate(m.backbone.model.layers):
                    hooks.append(layer.mlp.register_forward_hook(lambda module,args,out,i=i:mlps.__setitem__(i+1,out.detach().float())))
                original=m.normalize_target
                def capture(value,key,**kwargs):
                    self.assertFalse(value.requires_grad);self.assertEqual(value.dtype,torch.float32)
                    raw[key]=value.clone();return original(value,key,**kwargs)
                with patch.object(m,'normalize_target',side_effect=capture):
                    m._run_backbone(ctx,compute_auxiliary_losses=True,auxiliary_grad=True,collect_target_statistics=True,
                                    block_observer=lambda i,x:blocks.__setitem__(i,x.detach().float()))
                for hook in hooks:hook.remove()
                for layer in m.layers:
                    expected=sum(mlps[i] for i in range(layer,layer+4)) if arm in ('P7-mlp','P7-simple') else blocks[layer+3]-blocks[layer-1]
                    torch.testing.assert_close(raw[layer],expected,rtol=0,atol=0)

    def test_simple_matches_p4_estimator_and_has_no_relational_work(self):
        m=model('P7-simple');p4=model('P4-iso-4h');u=torch.randn(2,8,32,requires_grad=True)
        self.assertEqual(m.layers,p4.layers)
        self.assertEqual(m.settings.target_version,p4.settings.target_version)
        self.assertFalse(m.relational_proxy)
        self.assertFalse(m.increment_target)
        for layer in m.layers:
            head=m.heads[str(layer)];reference=p4.heads[str(layer)]
            self.assertEqual(set(dict(head.named_parameters())),
                             {'w1.weight','w2.weight','k_proj.weight','v_proj.weight','k_norm.weight'})
            for name in ('w1','w2'):
                torch.testing.assert_close(getattr(head,name).weight,getattr(reference,name).weight,rtol=0,atol=0)
            actual=head.estimate(u,None);expected=reference.estimates(u.detach())[0]
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            actual.square().sum().backward();expected.square().sum().backward()
            self.assertIsNone(u.grad)
            for name in ('w1','w2'):
                torch.testing.assert_close(getattr(head,name).weight.grad,getattr(reference,name).weight.grad,rtol=0,atol=0)
        for backend in ('sdpa','fa4'):
            simple=model('P7-simple',backend)
            with patch('deep_kv.proxy_memory.eligible_queries',side_effect=AssertionError('query sampling')), \
                 patch('deep_kv.proxy_memory.relational_losses',side_effect=AssertionError('relational loss')):
                out=simple(batch());objective(out).backward()
                self.assertNotIn('rel_sum',out)
                summary=summarize_proxy(out['statistics'].numpy(),simple)
                self.assertEqual(summary['cos_loss'],summary['aux_loss'])
                self.assertNotIn('rel_loss',summary)
        wrong=config();wrong.num_key_value_heads=4
        with self.assertRaisesRegex(ValueError,'four query heads'):
            ProxyModel.from_scratch(wrong,'P7-simple',consumer=2,deep_target=8)
        budget=compute_budget(config(),m.settings,8)['P7-simple']
        self.assertEqual(budget['relational_forward_macs_per_token'],0)

    def test_simple_p4_matching_targets_statistics_loss_and_full_depth_seed(self):
        # With both consumers disabled, equal backbones must produce equal
        # targets, normalization updates and estimator supervision.
        for backend in ('sdpa','fa4'):
            simple=model('P7-simple',backend);control=model('P4-iso-4h',backend)
            for step,expected in ((0,0.),(125,.05),(250,.1),(1000,.1)):
                self.assertEqual(simple.auxiliary_weight(step),expected)
                self.assertEqual(simple.auxiliary_weight(step),control.auxiliary_weight(step))
            with simple.without_proxy(),control.without_proxy():
                for phase in ('mean','variance'):
                    snapshots=[]
                    for m in (simple,control):
                        with torch.no_grad():
                            out=m(batch(),compute_auxiliary_losses=False,collect_target_statistics=True,statistics_mode=phase)
                        moments=tuple(out[key] for key in ('center_sums','center_squares','center_counts'))
                        snapshots.append(moments)
                        m.update_statistics(*moments,initialize=phase)
                    for a,b in zip(*snapshots):torch.testing.assert_close(a,b,rtol=2e-5,atol=1e-7)
                out=simple(batch(),collect_target_statistics=True)
                reference=control(batch(),collect_target_statistics=True)
                for key in ('lm_sum','aux_sum','center_sums','center_squares','center_counts','clip_counts'):
                    torch.testing.assert_close(out[key],reference[key],rtol=2e-5,atol=1e-6,msg=key)
                out['aux_sum'].backward();reference['aux_sum'].backward()
                for layer in simple.layers:
                    for name in ('w1','w2'):
                        a=getattr(simple.heads[str(layer)],name).weight.grad
                        b=getattr(control.heads[str(layer)],name).weight.grad
                        torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-5)
                for m,result in ((simple,out),(control,reference)):
                    m.update_statistics(*(result[k] for k in ('center_sums','center_squares','center_counts')))
                torch.testing.assert_close(simple.mu,control.mu,rtol=2e-5,atol=1e-7)
                torch.testing.assert_close(simple.sigma2,control.sigma2,rtol=2e-5,atol=1e-7)
        c=config();c.num_hidden_layers=28;c.layer_types=['full_attention']*28
        pair=[ProxyModel.from_scratch(c,arm,proxy_settings=ProxySettings(width=8,module_seed=77))
              for arm in ('P7-simple','P4-iso-4h')]
        self.assertEqual(pair[0].layers,tuple(range(2,25,2)))
        for layer in pair[0].layers:
            for name in ('w1','w2'):
                torch.testing.assert_close(getattr(pair[0].heads[str(layer)],name).weight,
                                           getattr(pair[1].heads[str(layer)],name).weight,rtol=0,atol=0)

    def test_simple_real_trainer_comparison_with_p4_control(self):
        arms=('A','P4-iso-4h','P7-simple')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);recipe=proxy_fixture(root);config().save_pretrained(root/'model')
            recipe.update(proxy_groups=2)
            for backend in ('sdpa','fa4'):
                destination=root/backend
                for arm in arms:
                    invoke(root,{**recipe,'attention_backend':backend,'arm':arm,'output_dir':str(destination/arm)})
                summary=report(destination,arms)
                self.assertEqual(set(summary['lm_loss']),set(arms))
                self.assertEqual(summary['proxy_targets']['P7-simple'],summary['proxy_targets']['P4-iso-4h'])
                configs=[json.loads((destination/arm/'train_config.json').read_text()) for arm in arms]
                self.assertEqual(len({c['train_fingerprint'] for c in configs}),1)
                # A differing auxiliary weight must not be reported as matched.
                path=destination/'P7-simple'/'train_config.json'
                changed=json.loads(path.read_text());changed['pilot']['proxy_lambda_max']=.3
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError,'Arms differ in configuration'):
                    report(destination,arms)
            path=root/'queue.json';path.write_text(json.dumps(recipe))
            queue=jobs(path,arms=arms,seeds=[42])
            runs=[job for job in queue['jobs'] if 'gpus' in job]
            self.assertEqual([job['argv'][job['argv'].index('--arm')+1] for job in runs],list(arms))
            self.assertTrue(all(job['gpus']==list(range(8)) for job in runs))

    def test_t7_sampling_rng_replay_empty_and_single_candidates(self):
        docs=torch.zeros(2,600,dtype=torch.long);docs[:,300:]=1
        state=torch.random.get_rng_state();a=MemoryPlan(docs,step=71,sequence_indices=[4,5])
        torch.testing.assert_close(state,torch.random.get_rng_state(),rtol=0,atol=0)
        b=MemoryPlan(docs,step=71,sequence_indices=[4,5]);c=MemoryPlan(docs,step=72,sequence_indices=[4,5])
        for x,y,z in zip(a.queries,b.queries,c.queries):
            self.assertEqual(len(x),256);self.assertEqual(len(x.unique()),256)
            self.assertNotIn(0,x);self.assertNotIn(300,x)
            self.assertTrue(torch.equal(x,y));self.assertFalse(torch.equal(x,z))
        self.assertFalse(torch.equal(a.queries[0],a.queries[1]))
        self.assertEqual(MemoryPlan(batch().segments).queries[0].tolist(),[1,2,4,5,6,7])
        for documents in (torch.arange(8)[None],torch.arange(4).repeat_interleave(2)[None]):
            pred=torch.randn(1,8,5,requires_grad=True);target=torch.randn_like(pred,requires_grad=True)
            rel,count=relational_losses(pred,target,MemoryPlan(documents))
            self.assertEqual(rel.item(),0);rel.sum().backward()
            self.assertIsNone(target.grad);self.assertTrue(torch.isfinite(pred.grad).all())

    def test_checkpoint_fa4_observer_bf16_and_mass(self):
        for arm in MEMORY_ARMS:
            reference=model(arm);expected=reference(batch());objective(expected).backward()
            for backend,checkpoint in (('sdpa',True),('fa4',False),('fa4',True)):
                m=model(arm,backend,checkpoint);observations=[];backwards=[]
                def observe(q,k,v,layout):
                    observations.append(layout)
                    for x in (q,k,v):
                        if x.requires_grad:x.register_hook(lambda gradient:backwards.append(True))
                m._fa4_observer=observe
                if backend=='fa4':
                    with patch.object(Context,'allowed',side_effect=AssertionError('dense FA4 mask')):
                        out=m(batch());objective(out).backward()
                    self.assertGreaterEqual(len(backwards),3*8)
                    if not checkpoint:
                        self.assertEqual(len(observations),8+len(m.layers))
                        self.assertEqual(len(backwards),3*(8+len(m.layers)))
                    self.assertTrue(any(layout[1]==10 for layout in observations))
                else:out=m(batch());objective(out).backward()
                for (name,p),(_,q) in zip(reference.named_parameters(),m.named_parameters()):
                    torch.testing.assert_close(p.grad,q.grad,rtol=3e-4,atol=3e-6,msg=name)
                torch.testing.assert_close(expected['statistics'],out['statistics'],rtol=1e-5,atol=1e-5)
                before={k:v.clone() for k,v in m.state_dict().items()}
                mass=m.eval().proxy_attention_mass(batch())
                self.assertTrue(((mass[:,0]>0)&(mass[:,0]<mass[:,1])).all())
                for name,value in before.items():torch.testing.assert_close(value,m.state_dict()[name],rtol=0,atol=0)
            m=model(arm)
            with torch.autocast('cpu',dtype=torch.bfloat16):out=m(batch());loss=objective(out)
            loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_accumulation_separate_token_and_query_denominators(self):
        ctx=batch();inputs=dict(input_ids=ctx.input_ids.repeat(2,1),labels=ctx.labels.repeat(2,1),
            attention_mask=ctx.valid.repeat(2,1),position_ids=ctx.position_ids.repeat(2,1),segments=ctx.segments.repeat(2,1))
        inputs['segments'][1]=torch.tensor([0,1,2,3,4,4,4,4]);inputs['position_ids'][1]=torch.tensor([0,0,0,0,0,1,2,3])
        for arm in ('P7','P7-simple'):
            with tempfile.TemporaryDirectory() as tmp:
                gradients=[];summaries=[]
                for micro in (2,1):
                    m=model(arm);m.mu_initialized.fill_(True)
                    trainer=ProxyTrainer(model=m,args=TrainingArguments(output_dir=tmp,use_cpu=True,report_to=[]))
                    trainer.is_in_train=True;trainer.state.global_step=300
                    samples=[{key:value[start:start+micro] for key,value in inputs.items()} for start in range(0,2,micro)]
                    counts=trainer._get_num_items_in_batch(samples,torch.device('cpu'))
                    self.assertEqual(counts.tolist(),[9,16] if arm=='P7-simple' else [9,16,9])
                    for sample in samples:trainer.compute_loss(m,sample,num_items_in_batch=counts).backward()
                    gradients.append({name:p.grad.clone() for name,p in m.named_parameters()})
                    summaries.append(summarize_proxy(trainer.step_totals.detach().numpy()[None],m))
                for name in gradients[0]:torch.testing.assert_close(gradients[0][name],gradients[1][name],rtol=2e-4,atol=2e-6,msg=name)
                for key in (('aux_loss','cos_loss') if arm=='P7-simple' else ('aux_loss','cos_loss','rel_loss')):
                    self.assertAlmostEqual(summaries[0][key],summaries[1][key],places=5)

    def test_real_trainer_resume_eval_reports_and_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);recipe=proxy_fixture(root);config().save_pretrained(root/'model')
            recipe.update(proxy_groups=2)
            for backend in ('sdpa','fa4'):
                destination=root/backend
                for arm in ('A',)+MEMORY_ARMS:
                    args={**recipe,'attention_backend':backend,'arm':arm,'output_dir':str(destination/arm)}
                    invoke(root,{**args,'stop_after':2});invoke(root,args)
                    invoke(root,{**args,'output_dir':str(destination/('full-'+arm))})
                    resumed=load_file(destination/arm/'model.safetensors');full=load_file(destination/('full-'+arm)/'model.safetensors')
                    for name in full:torch.testing.assert_close(resumed[name],full[name],rtol=0,atol=0,msg=name)
                    result=json.loads((destination/arm/'result.json').read_text())
                    if arm=='A':continue
                    self.assertEqual(result['proxy']['target']['target_version'],'p4p6-r1' if arm=='P7-simple' else 'p7-r1')
                    self.assertIn('eval_proxy_layer_2_attention_mass',result['evaluation'])
                    history=json.loads((destination/arm/'trainer_state.json').read_text())['log_history']
                    self.assertEqual(any('step_proxy_layer_2_rel_loss' in row for row in history),arm!='P7-simple')
                    if arm=='P7-simple':
                        self.assertNotIn('relational',result['proxy']['target'])
                        self.assertNotIn('eval_rel_loss',result['evaluation'])
                    with self.assertRaisesRegex(ValueError,'Resume configuration'):
                        invoke(root,{**args,'attention_backend':'sdpa' if backend=='fa4' else 'fa4'})
                report(destination,('A',)+MEMORY_ARMS)
                # Reject a mislabeled simple arm even if result/config agree.
                path=destination/'P7-simple'/'train_config.json'
                result_path=destination/'P7-simple'/'result.json'
                original_config=path.read_text();original_result=result_path.read_text()
                for change in ({'relational':{'weight':.5}}, {'quantity':'four_block_increment'}):
                    changed=json.loads(original_config);changed_result=json.loads(original_result)
                    changed['proxy_target'].update(change)
                    changed_result['proxy']['target'].update(change)
                    path.write_text(json.dumps(changed));result_path.write_text(json.dumps(changed_result))
                    with self.assertRaisesRegex(ValueError,'target metadata'):
                        report(destination,('A','P7-simple'))
                path.write_text(original_config);result_path.write_text(original_result)
            path=root/'queue.json';path.write_text(json.dumps(recipe))
            queue=jobs(path,arms=['A',*MEMORY_ARMS],seeds=[42])
            runs=[job for job in queue['jobs'] if 'gpus' in job]
            self.assertEqual(len(runs),1+len(MEMORY_ARMS));self.assertTrue(all(job['gpus']==list(range(8)) for job in runs))

    def test_budget_parameter_counts(self):
        c=config();settings=ProxySettings(width=8,features=9,chunk_size=2)
        budget=compute_budget(c,settings,8)
        for arm in MEMORY_ARMS:
            m=model(arm)
            self.assertEqual(sum(p.numel() for p in m.heads.parameters()),budget[arm]['extra_parameters'])

    def test_batched_relational_matches_loop_with_ragged_and_empty_queries(self):
        torch.manual_seed(815)
        documents=torch.stack((torch.zeros(300,dtype=torch.long),torch.arange(300)//150,
                               torch.arange(300),torch.arange(300)//2,torch.arange(300).clamp_min(3)))
        plan=MemoryPlan(documents,step=97,sequence_indices=list(range(5)))
        for dtype in (torch.float32,torch.bfloat16):
            pred=torch.randn(5,300,32,dtype=dtype,requires_grad=True)
            target=torch.randn_like(pred,requires_grad=True)
            a,n=relational_losses(pred,target,plan);b,m=loop_relational_reference(pred,target,plan)
            torch.testing.assert_close(a,b,rtol=2e-6,atol=2e-5)
            torch.testing.assert_close(n,m,rtol=0,atol=0)
            ga=torch.autograd.grad(a.sum(),pred)[0];gb=torch.autograd.grad(b.sum(),pred)[0]
            if dtype==torch.float32:
                torch.testing.assert_close(ga,gb,rtol=2e-4,atol=2e-6)
            else:
                # Test the FP32 derivative before the BF16 leaf cast as well:
                # tiny FP32 reduction differences can cross a BF16 midpoint.
                full=pred.detach().float().requires_grad_()
                fa,_=relational_losses(full,target,plan);fb,_=loop_relational_reference(full,target,plan)
                torch.testing.assert_close(torch.autograd.grad(fa.sum(),full)[0],
                    torch.autograd.grad(fb.sum(),full)[0],rtol=2e-4,atol=2e-6)
                down=torch.nextafter(gb,torch.full_like(gb,-torch.inf))
                up=torch.nextafter(gb,torch.full_like(gb,torch.inf))
                self.assertTrue(((ga>=down)&(ga<=up)).all())
            self.assertIsNone(target.grad);self.assertEqual(ga[2:4].count_nonzero(),0)
            self.assertTrue(torch.isfinite(ga).all())
        first=plan.relational_layout()
        self.assertIs(first,plan.relational_layout())
        self.assertEqual(n.tolist(),[256,256,0,150,3])
        indices,valid,blocked,_=first
        allowed=~blocked
        self.assertTrue(allowed.any(-1).all())
        for row in range(5):
            for slot in valid[row].nonzero().flatten():
                i=int(indices[row,slot])
                expected=(torch.arange(300)<i)&(documents[row]==documents[row,i])
                torch.testing.assert_close(allowed[row,slot],expected,rtol=0,atol=0)
        zero=torch.zeros(5,300,32,requires_grad=True)
        loss,_=relational_losses(zero,torch.zeros_like(zero),plan)
        loss.sum().backward();self.assertTrue(torch.isfinite(zero.grad).all())

    def test_optimized_full_model_gradients_and_sampling_skip(self):
        ctx=batch();rows=[]
        for segments in ([0]*3+[1]*5,list(range(8))):
            rows.append(dict(input_ids=ctx.input_ids[0].tolist(),labels=ctx.labels[0].tolist(),
                             attention_mask=[1]*8,segments=segments))
        ctx=ProxyTrainer.context(isolated_data_collator(rows))
        for arm in MEMORY_ARMS:
            for backend in ('sdpa','fa4'):
                a=model(arm,backend);b=model(arm,backend)
                actual=a(ctx);objective(actual).backward()
                with patch('deep_kv.proxy_memory.relational_losses',side_effect=loop_relational_reference):
                    expected=b(ctx);objective(expected).backward()
                torch.testing.assert_close(actual['statistics'],expected['statistics'],rtol=2e-6,atol=1e-5)
                for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
                    torch.testing.assert_close(p.grad,q.grad,rtol=3e-4,atol=3e-6,msg=name)
                with patch('deep_kv.proxy_memory.eligible_queries',side_effect=AssertionError('unneeded query sampling')):
                    a(ctx,compute_auxiliary_losses=False,collect_target_statistics=True,statistics_mode='mean')
                    a.eval().proxy_attention_mass(ctx)

    def test_midrun_evaluation_reports_cosine_trigger_and_mass(self):
        ctx=batch()
        row=dict(input_ids=ctx.input_ids[0].tolist(),labels=ctx.labels[0].tolist(),
                 attention_mask=[1]*8,segments=ctx.segments[0].tolist())
        with tempfile.TemporaryDirectory() as tmp:
            m=model();m.mu_initialized.fill_(True)
            trainer=ProxyTrainer(model=m,args=TrainingArguments(output_dir=tmp,use_cpu=True,report_to=[],remove_unused_columns=False),
                eval_dataset=Dataset.from_list([row]),data_collator=isolated_data_collator)
            trainer.is_in_train=True;trainer.state.global_step=1000
            callback=ProxyCallback(2500,8);callback.trainer=trainer
            control=TrainerControl();callback.on_step_begin(trainer.args,trainer.state,control)
            trainer.compute_loss(m,isolated_data_collator([row]))
            callback.on_step_end(trainer.args,trainer.state,control)
            self.assertTrue(control.should_evaluate)
            trainer.is_in_train=False
            metrics=trainer.evaluate()
            mean=sum(metrics[f'eval_proxy_layer_{layer}_cosine_0'] for layer in m.layers)/len(m.layers)
            self.assertEqual(metrics['eval_p7_mean_cosine'],mean)
            self.assertEqual(metrics['eval_p7_low_cosine_trigger'],mean<.3)
            self.assertIn('eval_proxy_layer_2_attention_mass',metrics)

    def test_two_rank_trainer_zero_lambda_and_query_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);recipe=proxy_fixture(root,world=2);config().save_pretrained(root/'model')
            for arm in MEMORY_ARMS:
                args={**recipe,'proxy_groups':2,'arm':arm,'output_dir':str(root/arm),
                      'ddp_backend':'gloo','ddp_find_unused_parameters':False,'logging_steps':2}
                path=root/'ddp.json';path.write_text(json.dumps(args))
                with (root/'worker.log').open('w') as output:
                    result=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                        'train.py',str(path)],cwd=Path(__file__).resolve().parents[1],
                        env={**os.environ,'OMP_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''},
                        stdout=output,stderr=subprocess.STDOUT,timeout=180)
                self.assertEqual(result.returncode,0,(root/'worker.log').read_text()[-18000:])
                result=json.loads((root/arm/'result.json').read_text())
                self.assertEqual(result['global_step'],3)
                self.assertTrue(torch.isfinite(torch.tensor(result['evaluation']['eval_aux_loss'])))

    def test_long_sequence_query_sampling_exact_trainer_resume(self):
        # Exercise the random subset branch (>256 eligible queries), not only
        # tiny contexts where every eligible query is selected.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);recipe=proxy_fixture(root)
            c=config();c.max_position_embeddings=512;c.save_pretrained(root/'model')
            tokenizer=AutoTokenizer.from_pretrained(root/'model');tokenizer.model_max_length=512
            tokenizer.save_pretrained(root/'model')
            for split,rows in (('long_train',16),('long_eval',4)):
                Dataset.from_dict({'text':[' '.join(str((i+j)%30) for j in range(299)) for i in range(rows)]}).save_to_disk(root/split)
            recipe.update(arm='P7',proxy_groups=2,block_size=300,data_dir=str(root/'long_train'),
                eval_data_dir=str(root/'long_eval'),eval_rows=3,monitor_rows=1,
                per_device_train_batch_size=1,per_device_eval_batch_size=1,
                checkpoint_layers=False,checkpoint_aux=False,checkpoint_lm=False,lm_chunk=300)
            invoke(root,{**recipe,'output_dir':str(root/'resume'),'stop_after':2})
            invoke(root,{**recipe,'output_dir':str(root/'resume')})
            invoke(root,{**recipe,'output_dir':str(root/'full')})
            a=load_file(root/'resume/model.safetensors');b=load_file(root/'full/model.safetensors')
            for name in a:torch.testing.assert_close(a[name],b[name],rtol=0,atol=0,msg=name)
            left=torch.load(root/'resume/checkpoint-3/optimizer.pt',weights_only=True)
            right=torch.load(root/'full/checkpoint-3/optimizer.pt',weights_only=True)
            self.assertEqual(left['param_groups'],right['param_groups'])
            for index,state in left['state'].items():
                for key,value in state.items():torch.testing.assert_close(value,right['state'][index][key],rtol=0,atol=0)
            self.assertEqual(torch.load(root/'resume/checkpoint-3/scheduler.pt',weights_only=True),
                             torch.load(root/'full/checkpoint-3/scheduler.pt',weights_only=True))


if __name__=='__main__':unittest.main()
