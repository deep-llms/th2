"""Nine follow-ups: independent objectives, delayed state, and real Trainer tests."""
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

from deep_kv import FOLLOWUP_ARMS, DELAYED_ARMS, P6_CONSUMER_ARMS, PROXY_DELAY_STEPS
from deep_kv import followup_metadata, anticipatory_target_metadata
from deep_kv.proxy import ProxyModel, ProxySettings, compute_budget
from tests.test_proxy_heads import cfg,batch,objective


def config():
    c=cfg();c.num_hidden_layers=12;c.layer_types=['full_attention']*12
    c.num_attention_heads=8;c.num_key_value_heads=4
    return c


def model(arm,checkpoint=False,backend='sdpa',**kw):
    return ProxyModel.from_scratch(config(),arm,consumer=2,deep_target=12,
        proxy_settings=ProxySettings(groups=2,width=8,features=9,chunk_size=2,**kw),
        checkpoint_layers=checkpoint,checkpoint_aux=checkpoint,checkpoint_lm=checkpoint,
        attention_backend=backend,lm_chunk=3)


def parent(arm):
    if arm.startswith('P7'):return 'P7-simple'
    if '-sparse' in arm:return 'P6-iso-sparse'
    return 'P6-iso'


def active(m):
    if m.arm in DELAYED_ARMS:m.set_training_step(PROXY_DELAY_STEPS+250)
    if m.extensions:
        from tests.test_proxy_extensions import activate
        activate(m)


def fixture(root,world=1):
    from tests.test_proxy_training import proxy_fixture
    recipe=proxy_fixture(root,world=world);config().save_pretrained(root/'model')
    recipe.update(proxy_groups=2,num_hidden_layers=12,deep_target=12,
        checkpoint_layers=False,checkpoint_lm=False,checkpoint_aux=False,proxy_sync_optimization=True,
        save_total_limit=0)
    return recipe


class Screen24Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_all_24_queue_and_parameter_contracts(self):
        from scripts.proxy_12layer_queue import ALL_SCREEN_ARMS,make
        self.assertEqual(len(ALL_SCREEN_ARMS),24);self.assertEqual(len(set(ALL_SCREEN_ARMS)),24)
        with tempfile.TemporaryDirectory() as tmp:
            items=make('proxy_heads.12layer.b200.json',Path(tmp)/'jobs.json',all_arms=True)
            gpu=[j for j in items if 'gpus' in j]
            self.assertEqual(len(gpu),24)
            self.assertEqual([j['argv'][j['argv'].index('--arm')+1] for j in gpu],list(ALL_SCREEN_ARMS))
            self.assertTrue(all(j['gpus']==list(range(8)) for j in gpu))
            self.assertEqual(len(items),50)  # Train+validate per arm, two reports.
        for arm in FOLLOWUP_ARMS:
            m=model(arm);p=model(parent(arm))
            for name,value in p.state_dict().items():torch.testing.assert_close(m.state_dict()[name],value,rtol=0,atol=0)
            if arm in DELAYED_ARMS:m.set_training_step(PROXY_DELAY_STEPS)
            torch.testing.assert_close(m(batch())['lm_sum'],p(batch())['lm_sum'],rtol=0,atol=0,msg=arm)
            budget=compute_budget(config(),m.settings,8)[arm]
            self.assertEqual(budget['extra_parameters'],sum(p.numel() for p in m.heads.parameters())+sum(p.numel() for p in m.extensions.parameters()))
        dense=model('P6-iso-deep');sparse=model('P6-iso-sparse-deep')
        for layer in sparse.layers:
            for name,value in sparse.extensions[str(layer)].state_dict().items():
                torch.testing.assert_close(value,dense.extensions[str(layer)].state_dict()[name],rtol=0,atol=0)

    def test_target_routes_against_independent_graph(self):
        for arm in ('P6-iso-sparse-targetgrad','P7-simple-targetgrad'):
            actual=model(arm);ref=model(parent(arm));ctx=batch();mlps={};sources={};handles=[]
            for i,block in enumerate(ref.backbone.model.layers,1):
                handles.append(block.mlp.register_forward_hook(lambda m,a,o,i=i:mlps.__setitem__(i,o)))
                if i in ref.layers:
                    def capture(m,a,o,i=i):
                        if i not in sources:sources[i]=o
                    handles.append(block.input_layernorm.register_forward_hook(capture))
            ref(ctx,compute_auxiliary_losses=False);losses=[]
            for layer in ref.layers:
                h=ref.heads[str(layer)];pred=h.w2(F.silu(h.w1(sources[layer].detach())))
                target=sum(mlps[i].float() for i in range(layer,layer+4))
                j=ref.mean_index[layer];v=ref.sigma2[j]
                truth=((target-ref.mu[j])*torch.rsqrt(v.clamp_min(.01*v.median())+1e-6)).clamp(-10,10)
                losses.append((1-F.cosine_similarity(pred.float(),truth,dim=-1,eps=1e-6)).sum())
            expected=torch.stack(losses).mean();out=actual(ctx,collect_target_statistics=True)
            torch.testing.assert_close(out['aux_sum'],expected,rtol=1e-6,atol=1e-6)
            out['aux_sum'].backward();expected.backward()
            self.assertFalse(out['center_sums'].requires_grad)
            for (n,p),(_,q) in zip(actual.named_parameters(),ref.named_parameters()):
                self.assertEqual(p.grad is None,q.grad is None,n)
                if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=3e-5,atol=2e-6,msg=arm+'/'+n)
            for h in handles:h.remove()

    def test_p7_lm_route_and_frozen_auxiliary_backbone(self):
        a,b=model('P7-simple'),model('P7-simple-lm')
        for m in (a,b):m(batch())['lm_sum'].backward()
        self.assertIsNone(a.heads['2'].w1.weight.grad)
        self.assertGreater(float(b.heads['2'].w1.weight.grad.abs().sum()),0)
        b.zero_grad(set_to_none=True);b(batch())['aux_sum'].backward()
        self.assertTrue(all(p.grad is None for p in b.backbone.parameters()))
        self.assertIsNone(b.heads['2'].k_proj.weight.grad)

    def test_consumer_objective_against_copied_frozen_decoder(self):
        # The reference copies blocks and disables their parameters. It does not
        # use the production functional consumer wrapper or its loss helper.
        for arm in P6_CONSUMER_ARMS:
            m=model(arm);ref=model('P6-iso');ctx=batch();mlps={};inputs={};handles=[]
            for i,block in enumerate(ref.backbone.model.layers,1):
                handles.append(block.mlp.register_forward_hook(lambda m,a,o,i=i:mlps.__setitem__(i,o.detach())))
                if i in ref.layers:
                    def capture(mod,args,i=i):
                        if i not in inputs:inputs[i]=args[0].detach()
                    handles.append(block.input_layernorm.register_forward_pre_hook(capture))
            ref(ctx,compute_auxiliary_losses=False)
            for h in handles:h.remove()
            terms=[];effects=[]
            for layer in ref.layers:
                h=inputs[layer];block=copy.deepcopy(ref.backbone.model.layers[layer-1]);block.requires_grad_(False)
                head=ref.heads[str(layer)]
                pred=head.w2(F.silu(head.w1(block.input_layernorm(h))))
                raw=sum(mlps[i].float() for i in range(layer,layer+4))
                truth=ref.normalize_target(raw,layer)
                rotary=ref.backbone.model.rotary_emb(h,ctx.position_ids)
                def response(message):
                    norm=message.float()*torch.rsqrt(message.float().square().mean(-1,keepdim=True)+1e-6)
                    x=h+(head.alpha.detach()[0]*norm*torch.sqrt(h.float().square().mean(-1,keepdim=True)+1e-6)).to(h.dtype)
                    u=block.input_layernorm(x);attn=ref.attention(block,u,None,ctx.allowed(),rotary)
                    res=x+block.self_attn.o_proj(attn.reshape(*u.shape[:-1],-1))
                    return res+block.mlp(block.post_attention_layernorm(res))
                teacher=response(truth).detach();student=response(pred)
                effect=((student.float()-teacher.float()).square().mean(-1)/(teacher.float().square().mean(-1)+1e-6)).sum()
                effects.append(effect.detach())
                cosine=(1-F.cosine_similarity(pred.float(),truth,dim=-1,eps=1e-6)).sum()
                terms.append(effect+(cosine if arm.endswith('consumer') else 0))
            expected=torch.stack(terms).mean();out=m(ctx)
            torch.testing.assert_close(out['aux_sum'],expected,rtol=2e-6,atol=2e-6)
            from deep_kv.proxy_training import summarize_proxy
            metrics=summarize_proxy(out['statistics'].detach().numpy(),m)
            self.assertAlmostEqual(metrics['consumer_effect_loss'],float(torch.stack(effects).mean()/out['aux_count']),places=6)
            out['aux_sum'].backward();expected.backward()
            for (n,p),(_,q) in zip(m.named_parameters(),ref.named_parameters()):
                self.assertEqual(p.grad is None,q.grad is None,n)
                if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=2e-5,atol=2e-6,msg=n)
            self.assertTrue(all(p.grad is None for p in m.backbone.parameters()))
            self.assertTrue(all(h.alpha.grad is None for h in m.heads.values()))
            # Functional substitution must restore the original parameter objects.
            self.assertTrue(all(isinstance(p,torch.nn.Parameter) and p.requires_grad for p in m.backbone.parameters()))

    def test_active_bfloat16_checkpoint_and_recompute(self):
        for arm in FOLLOWUP_ARMS:
            for bf16 in (False,True):
                ref=model(arm);active(ref)
                for m in (ref,model(arm,checkpoint=True),model(arm,aux_recompute=True) if not arm.startswith('P7') else model(arm)):
                    m.load_state_dict(ref.state_dict(),strict=True)
                    with torch.autocast('cpu',dtype=torch.bfloat16,enabled=bf16):out=m(batch());loss=objective(m,out)
                    loss.backward()
                    if m is ref:expected=out
                    else:
                        for key in ('lm_sum','aux_sum'):torch.testing.assert_close(out[key],expected[key],rtol=0,atol=0,msg=arm)
                        for (n,p),(_,q) in zip(m.named_parameters(),ref.named_parameters()):
                            self.assertEqual(p.grad is None,q.grad is None,n)
                            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=.03 if bf16 else 3e-5,atol=2e-6,msg=arm+'/'+n)

    def test_compiled_predictor_functions(self):
        compile_fn=torch.compile
        for arm in FOLLOWUP_ARMS:
            if arm.startswith('P7'):continue  # P7 rejects estimator compilation.
            eager=model(arm);active(eager)
            with patch('torch.compile',side_effect=lambda fn:compile_fn(fn,backend='aot_eager',fullgraph=True)):
                compiled=model(arm,compile_estimator=True)
            compiled.load_state_dict(eager.state_dict())
            for m in (eager,compiled):objective(m,m(batch())).backward()
            for (n,p),(_,q) in zip(eager.named_parameters(),compiled.named_parameters()):
                self.assertEqual(p.grad is None,q.grad is None,n)
                if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=2e-5,atol=2e-7,msg=arm+'/'+n)

    def test_fa4_reference_gradients_and_consumer_call_counts(self):
        from tests.test_fa4_baseline import reference_kernel
        with patch('deep_kv.fa4.load_kernel',return_value=(reference_kernel,{'version':'CPU-reference'})):
            for arm in FOLLOWUP_ARMS:
                a,b=model(arm),model(arm,backend='fa4');active(a);b.load_state_dict(a.state_dict())
                calls=[]
                b._fa4_observer=lambda q,k,v,layout:calls.append(q.requires_grad)
                x,y=a(batch()),b(batch())
                objective(a,x).backward();objective(b,y).backward()
                for key in ('lm_sum','aux_sum'):torch.testing.assert_close(x[key],y[key],rtol=3e-5,atol=3e-6,msg=arm)
                for (n,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
                    self.assertEqual(p.grad is None,q.grad is None,n)
                    if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=4e-4,atol=3e-6,msg=arm+'/'+n)
                extra=len(b.layers)*(2 if arm in P6_CONSUMER_ARMS else 1 if arm.startswith('P7') else 0)
                self.assertEqual(len(calls),12+extra)
                self.assertEqual(sum(calls),len(calls)-(len(b.layers) if arm in P6_CONSUMER_ARMS else 0))

    def test_document_isolation_and_causality(self):
        for arm in FOLLOWUP_ARMS:
            m=model(arm);active(m);ctx=batch();h=m.hidden_states(ctx)[0]
            other=copy.deepcopy(ctx);other.input_ids[:,:3]=20
            torch.testing.assert_close(m.hidden_states(other)[0][:,3:],h[:,3:],rtol=0,atol=0)
            other=copy.deepcopy(ctx);other.input_ids[:,6:]=20
            torch.testing.assert_close(m.hidden_states(other)[0][:,:6],h[:,:6],rtol=0,atol=0)
            with m.without_proxy():torch.testing.assert_close(m.hidden_states(ctx)[0],model('A').hidden_states(ctx)[0],rtol=0,atol=0)

    def test_delay_boundaries_and_standalone_restore(self):
        baseline=model('A');ctx=batch()
        for arm in DELAYED_ARMS:
            m=model(arm)
            for step in (0,499,500,501,750):
                m.set_training_step(step)
                self.assertEqual(m.proxy_enabled,step>=500)
                self.assertAlmostEqual(m.auxiliary_weight(step),.1*max(0,min((step-500)/250,1)))
                loaded=model(arm);loaded.load_state_dict(m.state_dict(),strict=True)
                self.assertEqual(loaded.proxy_enabled,m.proxy_enabled)
                expected=baseline if step<500 else model(parent(arm))
                torch.testing.assert_close(loaded.hidden_states(ctx)[0],expected.hidden_states(ctx)[0],rtol=0,atol=0)
                if arm.startswith('P7') and step<500:
                    mass=loaded.proxy_attention_mass(ctx)
                    self.assertEqual(tuple(mass.shape),(4,2));self.assertEqual(float(mass[:,0].sum()),0)
                    self.assertTrue(bool((mass[:,1]>0).all()))

    def test_global_denominators_for_all_followups(self):
        from transformers import TrainingArguments
        from deep_kv.proxy_training import ProxyTrainer
        from deep_kv.packing import isolated_data_collator
        c=batch();row=dict(input_ids=c.input_ids[0].tolist(),labels=c.labels[0].tolist(),attention_mask=[1]*8)
        rows=[{**row,'segments':s} for s in ([0]*3+[1]*5,[0]*4+[1]*4,[0]+[1]*2+[2]*5)]
        micro=[isolated_data_collator([row]) for row in rows]
        for arm in FOLLOWUP_ARMS:
            pair=[model(arm),model(arm)]
            for m in pair:active(m)
            with tempfile.TemporaryDirectory() as tmp:
                trainers=[ProxyTrainer(model=m,args=TrainingArguments(output_dir=tmp,use_cpu=True,report_to=[])) for m in pair]
                for t in trainers:t.is_in_train=True;t.state.global_step=750
                counts=trainers[0]._get_num_items_in_batch(micro,torch.device('cpu'))
                for inputs in micro:trainers[0].compute_loss(pair[0],inputs,num_items_in_batch=counts).backward()
                trainers[1].compute_loss(pair[1],isolated_data_collator(rows),num_items_in_batch=counts).backward()
            for (n,p),(_,q) in zip(pair[0].named_parameters(),pair[1].named_parameters()):
                self.assertEqual(p.grad is None,q.grad is None,n)
                if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=3e-5,atol=3e-7,msg=arm+'/'+n)

    def test_consumer_gradient_does_not_cross_documents(self):
        from deep_kv.proxy_estimators import consumer_effect_loss
        for backend in ('sdpa','fa4'):
            from tests.test_fa4_baseline import reference_kernel
            with patch('deep_kv.fa4.load_kernel',return_value=(reference_kernel,{'version':'CPU-reference'})):
                m=model('P6-iso-consumer',backend=backend);ctx=batch()
                hidden=torch.randn(1,8,32,requires_grad=True);pred=torch.randn_like(hidden,requires_grad=True)
                target=torch.randn_like(hidden,requires_grad=True)
                rotary=m.backbone.model.rotary_emb(hidden,ctx.position_ids)
                from deep_kv.fa4 import document_layout
                mask=document_layout(ctx) if backend=='fa4' else ctx.allowed()
                valid=ctx.valid.clone();valid[:,:3]=False
                loss=consumer_effect_loss(m.backbone.model.layers[1],m.attention,hidden,pred,target,
                    m.heads['2'].alpha,mask,rotary,valid).sum()
                loss.backward()
                self.assertIsNone(hidden.grad);self.assertIsNone(target.grad)
                self.assertEqual(int(pred.grad[:,:3].count_nonzero()),0)
                self.assertGreater(float(pred.grad[:,3:].abs().sum()),0)
                self.assertTrue(all(p.grad is None for p in m.parameters()))

    def test_trainer_resume_across_introduction_and_reports(self):
        from tests.test_train import invoke
        from safetensors.torch import load_file
        from eval.models import load_checkpoint
        from deep_kv.report import report
        # Bring the exact production transition into a three-update CPU fixture.
        with patch('deep_kv.proxy.PROXY_DELAY_STEPS',1),patch('deep_kv.proxy_training.PROXY_DELAY_STEPS',1),tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);recipe=fixture(root)
            for arm in ('A','P6-iso','P6-iso-sparse','P7-simple')+FOLLOWUP_ARMS:
                args={**recipe,'arm':arm,'output_dir':str(root/arm)}
                invoke(root,{**args,'stop_after':1});invoke(root,args)
                loaded,_,_=load_checkpoint(root/arm,device='cpu',attention_backend='sdpa')
                self.assertEqual(loaded.wrapped.arm,arm)
                config_json=json.loads((root/arm/'train_config.json').read_text())
                for key,value in followup_metadata(arm).items():self.assertEqual(config_json['proxy_target'][key],value)
                if arm in FOLLOWUP_ARMS:
                    invoke(root,{**args,'output_dir':str(root/('full-'+arm))})
                    resumed=load_file(root/arm/'model.safetensors');full=load_file(root/('full-'+arm)/'model.safetensors')
                    for k in resumed:torch.testing.assert_close(resumed[k],full[k],rtol=0,atol=0,msg=arm+'/'+k)
                    if arm in DELAYED_ARMS:
                        before=load_file(root/arm/'checkpoint-1/model.safetensors')
                        init=model(arm)
                        # Bootstrap statistics can change, dormant parameters cannot.
                        for n,p in init.heads.named_parameters():torch.testing.assert_close(before['heads.'+n],p,rtol=0,atol=0)
                        self.assertEqual(int(resumed['proxy_updates']),3)
                        self.assertTrue(loaded.wrapped.proxy_enabled)
                        optimizer=torch.load(root/arm/'checkpoint-1/optimizer.pt',weights_only=True)
                        self.assertEqual(len(optimizer['state']),len(list(init.backbone.parameters())))
            arms=('A','P6-iso','P6-iso-sparse','P7-simple')+FOLLOWUP_ARMS
            result=report(root,arms)
            self.assertIn('P7-simple-lm-P7-simple',result['nll_differences'])
            for arm in FOLLOWUP_ARMS:
                p=root/arm/'train_config.json';original=p.read_text();value=json.loads(original)
                key=next(iter(followup_metadata(arm)), 'extension')
                value['proxy_target'][key]={'tampered':True};p.write_text(json.dumps(value))
                with self.assertRaises(ValueError):report(root,arms)
                p.write_text(original)

    def test_dormant_trainer_matches_baseline_and_keeps_proxy_frozen(self):
        from tests.test_train import invoke
        from safetensors.torch import load_file
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);recipe=fixture(root)
            invoke(root,{**recipe,'arm':'A','output_dir':str(root/'A')})
            base=load_file(root/'A/model.safetensors')
            for arm in DELAYED_ARMS:
                invoke(root,{**recipe,'arm':arm,'output_dir':str(root/arm)})
                weights=load_file(root/arm/'model.safetensors');initial=model(arm)
                for name,value in base.items():
                    if name.startswith('backbone.'):
                        torch.testing.assert_close(weights[name],value,rtol=1e-6,atol=1e-8,msg=arm+'/'+name)
                for name,p in initial.heads.named_parameters():
                    torch.testing.assert_close(weights['heads.'+name],p,rtol=0,atol=0)
                optimizer=torch.load(root/arm/'checkpoint-3/optimizer.pt',weights_only=True)
                self.assertEqual(len(optimizer['state']),len(list(initial.backbone.parameters())))

    def test_two_rank_new_objectives_and_delay_transition(self):
        for arm in FOLLOWUP_ARMS:
            with self.subTest(arm=arm),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);recipe=fixture(root,world=2)
                recipe.update(arm=arm,ddp_backend='gloo',ddp_find_unused_parameters=False,logging_steps=2,eval_on_start=False)
                path=root/'recipe.json';path.write_text(json.dumps(recipe));log=root/'ddp.log'
                with log.open('w') as out:
                    p=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                        '--module','tests.test_proxy_screen24','--ddp-worker',str(path)],
                        cwd=Path(__file__).resolve().parents[1],env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1'},
                        stdout=out,stderr=subprocess.STDOUT,timeout=180)
                self.assertEqual(p.returncode,0,log.read_text()[-15000:])
                result=json.loads((Path(recipe['output_dir'])/'result.json').read_text())
                self.assertEqual(result['global_step'],3)


if __name__=='__main__':
    if '--ddp-worker' in sys.argv:
        import deep_kv.proxy as proxy
        import deep_kv.proxy_training as training
        proxy.PROXY_DELAY_STEPS=training.PROXY_DELAY_STEPS=1
        import train
        sys.argv=['train.py',sys.argv[-1]]
        train.main()
    else:unittest.main()
