"""P4/P5/P6 acceptance using the real model and HF Trainer on tiny CPU data."""
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
from safetensors.torch import load_file
from transformers import TrainingArguments
from deep_kv import ANTICIPATORY_ARMS
from deep_kv.proxy import ProxyModel, ProxySettings, compute_budget
from deep_kv.proxy_estimators import BlockMLP
from deep_kv.proxy_training import ProxyTrainer
from deep_kv.packing import isolated_data_collator
from deep_kv.report import report, comparable_config
from deep_kv.__main__ import jobs
from tests.test_proxy_heads import model, batch, cfg, settings, objective
from tests.test_proxy_training import proxy_fixture
from tests.test_train import invoke


class AnticipatoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_placement_budget_initialization_and_zero_gates(self):
        config=cfg();config.num_hidden_layers=28;config.layer_types=['full_attention']*28
        base=model('A');ctx=batch()
        for arm in ANTICIPATORY_ARMS:
            m=model(arm)
            full=ProxyModel.from_scratch(config,arm,proxy_settings=settings())
            self.assertEqual(full.layers,tuple(range(2,25,2)))
            for key,value in base.backbone.state_dict().items():
                torch.testing.assert_close(m.backbone.state_dict()[key],value,rtol=0,atol=0)
            with torch.no_grad():
                for head in m.heads.values():head.alpha.zero_()
                expected=base.backbone.lm_head(base.hidden_states(ctx)[0])
                actual=m.backbone.lm_head(m.hidden_states(ctx)[0])
            torch.testing.assert_close(actual,expected,rtol=0,atol=1e-5)
            other=model(arm,module_seed=99)
            self.assertTrue(all(torch.equal(v,other.backbone.state_dict()[k]) for k,v in m.backbone.state_dict().items()))
            self.assertFalse(torch.equal(m.heads['2'].w1.weight,other.heads['2'].w1.weight))
            changed=ProxyModel.from_scratch(cfg(),arm,seed=100,consumer=2,deep_target=8,proxy_settings=settings())
            torch.testing.assert_close(m.heads['2'].w1.weight,changed.heads['2'].w1.weight,rtol=0,atol=0)
        config.hidden_size=1024;config.intermediate_size=3072;config.head_dim=128
        config.num_attention_heads=16;config.num_key_value_heads=8
        budget=compute_budget(config,ProxySettings())
        self.assertEqual(budget['P4']['extra_macs_per_token'],6291456)
        self.assertEqual(budget['P6']['extra_macs_per_token'],6291456)
        self.assertEqual(budget['P5']['extra_macs_per_token'],8454144)

    def test_native_qk_and_all_values(self):
        for arm in ('P4','P4-iso','P5'):
            m=model(arm);ctx=batch();u=torch.randn(1,8,32);z=u+torch.randn_like(u)
            for index in m.layers:
                layer=m.backbone.model.layers[index-1];captured=[]
                rotary=m.backbone.model.rotary_emb(u,ctx.position_ids)
                def attention(module,q,k,v,mask,**kwargs):
                    captured.append((q,k,v))
                    return v.repeat_interleave(module.num_key_value_groups,dim=1).transpose(1,2),None
                with patch('deep_kv.proxy.ALL_ATTENTION_FUNCTIONS.get_interface',return_value=attention):
                    m.attention(layer,u,None,ctx.allowed(),rotary)
                    m.attention(layer,u,z,ctx.allowed(),rotary)
                for i in (0,1):torch.testing.assert_close(captured[0][i],captured[1][i],rtol=0,atol=0)
                expected=layer.self_attn.v_proj(z).view(1,8,2,8).transpose(1,2)
                torch.testing.assert_close(captured[1][2],expected,rtol=0,atol=0)
                self.assertTrue(all((captured[0][2][:,i]-expected[:,i]).abs().max()>0 for i in range(2)))

    def test_block_autograd_reference_fp32_and_bf16(self):
        for dtype in (torch.float32,torch.bfloat16):
            torch.manual_seed(19)
            originals=[torch.randn(*shape) for shape in ((2,7,32),(8,32),(32,8))]
            for lm_enabled in (False,True):
                results=[]
                for reference in (False,True):
                    x,w1,w2=[v.clone().requires_grad_() for v in originals]
                    with torch.autocast('cpu',dtype=dtype,enabled=dtype!=torch.float32):
                        if reference:
                            lm=torch.nn.functional.linear(torch.nn.functional.silu(torch.nn.functional.linear(x,w1)),w2)
                            aux=torch.nn.functional.linear(torch.nn.functional.silu(torch.nn.functional.linear(x.detach(),w1)),w2)
                        else:lm,aux=BlockMLP.apply(x,w1,w2)
                        loss=aux.float().square().mean()+(.3*lm.float().sin().mean() if lm_enabled else 0)
                    loss.backward();results.append((lm,aux,x.grad,w1.grad,w2.grad))
                for a,b in zip(*results):
                    self.assertEqual(a is None,b is None)
                    if a is not None:
                        error=(a-b).abs().max()/b.abs().max().clamp_min(1e-20)
                        self.assertLessEqual(float(error),1e-6 if dtype==torch.float32 else .01)

    def test_isolation_gradient_paths_and_stream_recurrence(self):
        for arm in ANTICIPATORY_ARMS:
            m=model(arm)
            for auxiliary in (False,True):
                m.zero_grad(set_to_none=True);out=m(batch());out['aux_sum' if auxiliary else 'lm_sum'].backward()
                for name,p in m.named_parameters():
                    nonzero=p.grad is not None and bool(p.grad.count_nonzero())
                    estimator=name.startswith('heads.') and not name.endswith('.alpha')
                    if auxiliary:self.assertEqual(nonzero,estimator,name)
                    elif estimator:self.assertEqual(nonzero,not m.settings.isolate_estimator,name)
                for h in m.heads.values():
                    if not auxiliary:self.assertGreater(h.alpha.grad.abs().sum(),0)
            if arm=='P5':
                m.zero_grad(set_to_none=True);u=torch.randn(1,8,32,requires_grad=True)
                _,state=m.heads['2'].stream_step(u,None)
                pred,_=m.heads['4'].stream_step(u,state);pred.square().sum().backward()
                self.assertIsNone(u.grad)
                self.assertGreater(m.heads['2'].w1.weight.grad.abs().sum(),0)
                self.assertIsNone(m.heads['2'].w2.weight.grad)

    def test_recompute_checkpoint_and_targets(self):
        for arm in ANTICIPATORY_ARMS:
            reference=model(arm);out=reference(batch(),collect_target_statistics=True);objective(reference,out).backward()
            for options in (dict(aux_recompute=True),dict(checkpoint=True)):
                m=model(arm,**options);mlps={};targets={};hooks=[]
                for i,layer in enumerate(m.backbone.model.layers,1):
                    hooks.append(layer.mlp.register_forward_hook(self.capture(mlps,i)))
                original=m.normalize_target
                def target(value,key,**kwargs):
                    self.assertFalse(value.requires_grad);targets[key]=value.clone()
                    return original(value,key,**kwargs)
                before={k:v.clone() for k,v in m.named_buffers()}
                with patch.object(m,'normalize_target',side_effect=target):actual=m(batch(),collect_target_statistics=True)
                for key,value in targets.items():
                    torch.testing.assert_close(value,sum(mlps[i].float() for i in range(key,key+4)),rtol=0,atol=0)
                objective(m,actual).backward()
                for h in hooks:h.remove()
                for key,value in m.named_buffers():torch.testing.assert_close(value,before[key],rtol=0,atol=0)
                for key in ('lm_sum','aux_sum','center_sums','center_squares','center_counts'):
                    torch.testing.assert_close(actual[key],out[key],rtol=0,atol=0)
                for (name,p),(_,q) in zip(m.named_parameters(),reference.named_parameters()):
                    self.assertEqual(p.grad is None,q.grad is None,name)
                    if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=1e-6,atol=2e-7,msg=arm+'/'+name)

    @staticmethod
    def capture(values,key):
        def hook(module,args,out):values[key]=out.detach().clone()
        return hook

    def test_document_isolation_targets_and_stream(self):
        for arm in ANTICIPATORY_ARMS:
            m=model(arm,checkpoint=True);ctx=batch();inputs=[];targets=[];streams=[]
            h=m.backbone.model.embed_tokens.register_forward_hook(lambda mod,args,out:inputs.append(out))
            original=m.layer_cosines;block=m.block
            def cosine(layer,u,estimates,raw,plan,valid,auxiliary_grad):
                targets.append(raw[:,3:].detach().clone());eligible=valid.clone();eligible[:,:3]=False
                return original(layer,u,estimates,raw,plan,eligible,auxiliary_grad)
            def observed(*args,**kw):
                result=block(*args,**kw)
                if arm=='P5':streams.append(None if result[-1] is None else result[-1][:,3:].detach().clone())
                return result
            ctx.labels[:,:3]=-100
            with patch.object(m,'layer_cosines',side_effect=cosine),patch.object(m,'block',side_effect=observed):
                out=m(ctx);first=targets.copy();first_streams=streams.copy()
                grad=torch.autograd.grad(out['lm_sum']+.1*out['aux_sum'],inputs[0])[0]
                self.assertEqual(grad[:,:3].count_nonzero(),0)
                targets.clear();streams.clear()
                changed=copy.deepcopy(ctx);changed.input_ids[:,:3]=20
                other=m(changed)
                torch.testing.assert_close(out['lm_sum'],other['lm_sum'],rtol=0,atol=0)
                for a,b in zip(first,targets):torch.testing.assert_close(a,b,rtol=0,atol=0)
                for a,b in zip(first_streams,streams):
                    if a is not None:torch.testing.assert_close(a,b,rtol=0,atol=0)
            h.remove()

    def test_p6_relative_scale(self):
        # Use production dimensions: tiny estimators can be below the specified epsilon.
        config=cfg();config.hidden_size=1024;config.intermediate_size=64
        m=ProxyModel.from_scratch(config,'P6',consumer=2,deep_target=8,proxy_settings=ProxySettings(width=256))
        ctx=batch();hidden=torch.randn(1,8,1024)*.02
        rotary=m.backbone.model.rotary_emb(hidden,ctx.position_ids)
        for index in m.layers:
            values=[];layer=m.backbone.model.layers[index-1]
            hook=layer.input_layernorm.register_forward_pre_hook(lambda mod,args:values.append(args[0]))
            m.block(index-1,hidden,ctx.allowed(),rotary,None);hook.remove()
            ratio=(values[1]-values[0]).square().mean(-1).sqrt()/values[0].square().mean(-1).sqrt()
            self.assertTrue(bool(((ratio>.09)&(ratio<.11)).all()))

    def test_real_trainer_resume_queue_and_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);recipe=proxy_fixture(root)
            for arm in ('A',)+ANTICIPATORY_ARMS:
                args={**recipe,'arm':arm,'output_dir':str(root/arm)}
                invoke(root,{**args,'stop_after':2});invoke(root,args)
                if arm!='A':
                    invoke(root,{**args,'output_dir':str(root/('full-'+arm))})
                    resumed=load_file(root/arm/'model.safetensors');full=load_file(root/('full-'+arm)/'model.safetensors')
                    for key in full:torch.testing.assert_close(resumed[key],full[key],rtol=0,atol=0,msg=arm+'/'+key)
                    saved=json.loads((root/arm/'train_config.json').read_text())
                    self.assertEqual(saved['pilot']['proxy_target_version'],'p4p6-r1')
                    logs=json.loads((root/arm/'trainer_state.json').read_text())['log_history']
                    self.assertTrue(any(row['step']==0 and 'proxy_layer_2_mean_abs_alpha_0' in row for row in logs))
                    for change in ({'proxy_module_seed':99},{'proxy_aux_recompute':True}):
                        with self.assertRaisesRegex(ValueError,'Resume configuration'):invoke(root,{**args,**change})
                    bad=copy.deepcopy(saved);bad['pilot']['proxy_alpha_init']=9
                    with self.assertRaises(ValueError):comparable_config(bad)
            result=report(root,('A',)+ANTICIPATORY_ARMS)
            self.assertIn('P4-iso-A',result['nll_differences'])
            path=root/'recipe.json';path.write_text(json.dumps(recipe))
            queue=jobs(path,arms=['P4-iso','P6'],seeds=[42])
            self.assertEqual(len(queue['jobs']),4)  # two trains, per-seed and seed comparisons

    def test_two_rank_trainer_zero_lambda_then_auxiliary(self):
        # Real DDP catches unused-estimator hooks at lambda=0 followed by >0.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);recipe=proxy_fixture(root,world=2)
            for arm in ('P4-iso','P5','P6'):
                args={**recipe,'arm':arm,'output_dir':str(root/arm),'ddp_backend':'gloo',
                      'ddp_find_unused_parameters':False,'logging_steps':2,'eval_on_start':False}
                path=root/'ddp.json';path.write_text(json.dumps(args))
                log=root/f'{arm}.log'
                with log.open('w') as handle:
                    result=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                        'train.py',str(path)],cwd=Path(__file__).resolve().parents[1],
                        env={**os.environ,'OMP_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''},
                        stdout=handle,stderr=subprocess.STDOUT,timeout=180)
                self.assertEqual(result.returncode,0,log.read_text()[-18000:])
                saved=json.loads((root/arm/'result.json').read_text())
                self.assertEqual(saved['global_step'],3)
                self.assertTrue(torch.isfinite(torch.tensor(saved['evaluation']['eval_aux_loss'])))

    def test_compiled_functions_keep_state_names_and_gradients(self):
        compile_fn=torch.compile
        for arm in ANTICIPATORY_ARMS:
            eager=model(arm)
            with patch('torch.compile',side_effect=lambda fn:compile_fn(fn,backend='aot_eager',fullgraph=True)):
                compiled=model(arm,compile_estimator=True)
            self.assertEqual(list(eager.state_dict()),list(compiled.state_dict()))
            for m in (eager,compiled):objective(m,m(batch())).backward()
            for (name,p),(_,q) in zip(eager.named_parameters(),compiled.named_parameters()):
                self.assertEqual(p.grad is None,q.grad is None,name)
                if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=2e-7,msg=name)

    def test_global_loss_denominators_across_microbatches(self):
        ctx=batch();row=dict(input_ids=ctx.input_ids[0].tolist(),labels=ctx.labels[0].tolist(),attention_mask=[1]*8)
        rows=[{**row,'segments':segments} for segments in ([0]*3+[1]*5,[0]*4+[1]*4,[0]+[1]*2+[2]*5)]
        micro=[isolated_data_collator([row]) for row in rows]
        for arm in ('P4-iso','P5','P6'):
            models=[model(arm),model(arm)]
            with tempfile.TemporaryDirectory() as tmp:
                trainers=[ProxyTrainer(model=m,args=TrainingArguments(output_dir=tmp,use_cpu=True,report_to=[])) for m in models]
                for t in trainers:t.is_in_train=True;t.state.global_step=250
                counts=trainers[0]._get_num_items_in_batch(micro,torch.device('cpu'))
                for inputs in micro:trainers[0].compute_loss(models[0],inputs,num_items_in_batch=counts).backward()
                trainers[1].compute_loss(models[1],isolated_data_collator(rows),num_items_in_batch=counts).backward()
                self.assertFalse(any(name.endswith('.alpha') for name in trainers[0].get_decay_parameter_names(models[0])))
            for (name,p),(_,q) in zip(models[0].named_parameters(),models[1].named_parameters()):
                self.assertEqual(p.grad is None,q.grad is None,name)
                if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=2e-5,atol=3e-7,msg=name)


if __name__=='__main__':unittest.main()
