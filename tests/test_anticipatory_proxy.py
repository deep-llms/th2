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

    def test_four_head_gqa_mapping_and_unchanged_parameters(self):
        config=cfg();config.num_attention_heads=16;config.num_key_value_heads=8
        ctx=batch();u=torch.randn(1,8,32);z=(u+torch.randn_like(u)).requires_grad_()
        for arm,parent in (('P4-4h','P4'),('P4-iso-4h','P4-iso')):
            m=ProxyModel.from_scratch(config,arm,consumer=2,deep_target=8,proxy_settings=settings())
            original=ProxyModel.from_scratch(config,parent,consumer=2,deep_target=8,proxy_settings=settings())
            self.assertEqual(m.value_groups,2)
            for key,value in original.state_dict().items():
                torch.testing.assert_close(m.state_dict()[key],value,rtol=0,atol=0)
            self.assertEqual(m.settings,original.settings)
            rotary=m.backbone.model.rotary_emb(u,ctx.position_ids)
            for index in m.layers:
                layer=m.backbone.model.layers[index-1];captured=[]
                def attention(module,q,k,v,mask,**kwargs):
                    captured.append((q,k,v))
                    return torch.nn.functional.scaled_dot_product_attention(q,k,v,
                        attn_mask=mask,enable_gqa=True,scale=module.scaling).transpose(1,2),None
                with patch('deep_kv.proxy.ALL_ATTENTION_FUNCTIONS.get_interface',return_value=attention):
                    native=m.attention(layer,u,None,ctx.allowed(),rotary)
                    partial=m.attention(layer,u,z,ctx.allowed(),rotary)
                for i in (0,1):torch.testing.assert_close(captured[0][i],captured[1][i],rtol=0,atol=0)
                torch.testing.assert_close(captured[0][2][:,:6],captured[1][2][:,:6],rtol=0,atol=0)
                expected=layer.self_attn.v_proj(z).view(1,8,8,8).transpose(1,2)
                torch.testing.assert_close(captured[1][2][:,6:],expected[:,6:],rtol=0,atol=0)
                torch.testing.assert_close(native[:,:,:12],partial[:,:,:12],rtol=0,atol=0)
                self.assertTrue(all((native[:,:,i]-partial[:,:,i]).abs().max()>0 for i in range(12,16)))
                grad=torch.autograd.grad(partial[:,:,:12].sum(),z,retain_graph=True)[0]
                self.assertEqual(grad.count_nonzero(),0)
                self.assertGreater(torch.autograd.grad(partial[:,:,12:].square().sum(),z)[0].abs().sum(),0)
            for component in ('lm_sum','aux_sum'):
                m.zero_grad(set_to_none=True);m(ctx)[component].backward()
                for name,p in m.named_parameters():
                    active=p.grad is not None and bool(p.grad.count_nonzero())
                    estimator=name.startswith('heads.') and not name.endswith('.alpha')
                    if component=='aux_sum':self.assertEqual(active,estimator,name)
                    elif estimator:self.assertEqual(active,not m.settings.isolate_estimator,name)
        config.num_key_value_heads=1
        with self.assertRaisesRegex(ValueError,'complete GQA groups'):
            ProxyModel.from_scratch(config,'P4-4h',consumer=2,deep_target=8,proxy_settings=settings())

    def test_four_head_real_trainer_and_resume_with_qwen_gqa_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);recipe=proxy_fixture(root)
            config=cfg();config.num_attention_heads=16;config.num_key_value_heads=8
            config.save_pretrained(root/'model')
            for arm in ('P4-4h','P4-iso-4h'):
                args={**recipe,'arm':arm,'output_dir':str(root/arm)}
                invoke(root,{**args,'stop_after':2})
                with self.assertRaisesRegex(ValueError,'Resume configuration'):
                    invoke(root,{**args,'arm':arm.removesuffix('-4h')})
                invoke(root,args);invoke(root,{**args,'output_dir':str(root/('full-'+arm))})
                resumed=load_file(root/arm/'model.safetensors');full=load_file(root/('full-'+arm)/'model.safetensors')
                for key in full:torch.testing.assert_close(resumed[key],full[key],rtol=0,atol=0)
            summary=report(root,('P4-4h','P4-iso-4h'))
            self.assertEqual(set(summary['lm_loss']),{'P4-4h','P4-iso-4h'})
            self.assertEqual(summary['nll_differences']['P4-iso-4h-P4-4h'],
                summary['lm_loss']['P4-iso-4h']-summary['lm_loss']['P4-4h'])
            path=root/'recipe.json';path.write_text(json.dumps(recipe))
            queue=jobs(path,arms=['P4-4h','P4-iso-4h'],seeds=[42])
            training=[job for job in queue['jobs'] if 'gpus' in job]
            self.assertEqual(len(training),2)
            self.assertTrue(all(job['gpus']==list(range(8)) for job in training))

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
            # A matching version string alone must not accept a different target.
            path=root/'P4-iso'/'train_config.json';saved=path.read_text()
            for key,value in (('bands',[2]),('quantity','deep_band_increment'),('lookahead',3),
                              ('normalization','none'),('epsilon',.1)):
                changed=json.loads(saved);changed['proxy_target'][key]=value
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError,'target metadata'):report(root,('A','P4-iso'))
            path.write_text(saved)
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

    def test_initialization_only_seed_keeps_consumed_data_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);recipe=proxy_fixture(root);seen={};fingerprints={};initial={}
            original=ProxyTrainer.training_step
            for name,seed,data_seed in (('base',42,42),('init',1042,42),('data',1042,1042)):
                seen[name]=[]
                def observe(trainer,wrapped,inputs,*args,**kwargs):
                    seen[name].append(inputs['input_ids'].clone())
                    if name not in initial:initial[name]=trainer.model.backbone.model.embed_tokens.weight.detach().clone()
                    return original(trainer,wrapped,inputs,*args,**kwargs)
                with patch.object(ProxyTrainer,'training_step',observe):
                    invoke(root,{**recipe,'seed':seed,'data_seed':data_seed,'arm':'A','output_dir':str(root/name)})
                fingerprints[name]=json.loads((root/name/'train_config.json').read_text())['train_fingerprint']
            self.assertEqual(fingerprints['base'],fingerprints['init'])
            self.assertNotEqual(fingerprints['base'],fingerprints['data'])
            torch.testing.assert_close(torch.cat(seen['base']),torch.cat(seen['init']),rtol=0,atol=0)
            self.assertFalse(torch.equal(torch.cat(seen['base']),torch.cat(seen['data'])))
            self.assertFalse(torch.equal(initial['base'],initial['init']))
            torch.testing.assert_close(initial['init'],initial['data'],rtol=0,atol=0)

    def test_mixed_precision_separate_loss_routes_with_compilation(self):
        compile_fn=torch.compile
        for arm in ANTICIPATORY_ARMS:
            for compiled in (False,True):
                with patch('torch.compile',side_effect=lambda fn:compile_fn(fn,backend='aot_eager',fullgraph=True)):
                    m=model(arm,checkpoint=True,compile_estimator=compiled)
                reference=model(arm,aux_recompute=True)
                for component in ('lm_sum','aux_sum'):
                    results=[]
                    for current in (m,reference):
                        current.zero_grad(set_to_none=True)
                        with torch.autocast('cpu',dtype=torch.bfloat16):out=current(batch())
                        out[component].backward();results.append(out)
                    for key in ('lm_sum','aux_sum'):
                        torch.testing.assert_close(results[0][key],results[1][key],rtol=0,atol=0)
                    for (name,p),(_,q) in zip(m.named_parameters(),reference.named_parameters()):
                        if q.grad is None or not bool(q.grad.count_nonzero()):
                            self.assertTrue(p.grad is None or not bool(p.grad.count_nonzero()),arm+'/'+name)
                        else:
                            self.assertIsNotNone(p.grad,arm+'/'+name)
                            error=(p.grad-q.grad).abs().max()/q.grad.abs().max()
                            self.assertLessEqual(float(error),.02,arm+'/'+name)


if __name__=='__main__':unittest.main()
