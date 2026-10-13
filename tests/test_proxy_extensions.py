"""Local CPU checks for paired P6 extensions, including nonzero learned paths."""
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

from deep_kv import P6_EXTENSION_ARMS, P6_ADAPTER_ARMS, P6_DEPTH_ARMS, P6_DEEP_ARMS
from deep_kv import anticipatory_target_metadata
from deep_kv.proxy import ProxyModel, compute_budget, resolve_proxy_settings
from tests.test_proxy_heads import model, batch, cfg, settings, objective


def activate(m):
    # A nonzero extension is essential: all six start at the parent function.
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(731)
        for extension in m.extensions.values():
            parameter = extension.weight if isinstance(extension, torch.nn.Linear) else extension.up.weight
            parameter.normal_(std=.02)


def reference(m, ctx):
    """Independent forward wiring, full depth history, direct target sums."""
    def norm(x):
        return (x.float()/torch.sqrt(x.float().square().mean(-1,keepdim=True)+1e-6)).to(x.dtype)
    def scale(x):
        return torch.sqrt(x.detach().float().square().mean(-1,keepdim=True)+1e-6)
    def adapter(module,x):
        return F.linear(F.silu(F.linear(norm(x),module.down.weight)),module.up.weight)
    h=m.backbone.model.embed_tokens(ctx.input_ids)
    history=[h]; pending={}; predictions={}; mlps={}
    rotary=m.backbone.model.rotary_emb(h,ctx.position_ids)
    for number,layer in enumerate(m.backbone.model.layers,1):
        if number in m.layers:
            head=m.heads[str(number)];u=layer.input_layernorm(h)
            projected=F.linear(u.detach(),head.w1.weight)
            if m.arm in P6_DEPTH_ARMS:
                x=h.detach() if m.arm.endswith('-current') else h.detach()-history[max(0,number-3)].detach()
                projected=projected+F.linear(norm(x),m.extensions[str(number)].weight)
            pred=F.linear(F.silu(projected),head.w2.weight)
            predictions[number]=pred
            if m.arm in P6_DEEP_ARMS:
                pending[number+3]=(number,u.detach() if m.arm.endswith('-shallow') else pred)
            delta=head.alpha[0]*norm(pred.detach())
            if m.arm in P6_ADAPTER_ARMS:
                delta=delta+adapter(m.extensions[str(number)],u.detach() if m.arm.endswith('-shallow') else pred.detach())
            h=h+(delta*scale(h)).to(h.dtype)
        u=layer.input_layernorm(h)
        attended=m.attention(layer,u,None,ctx.allowed(),rotary)
        residual=h+layer.self_attn.o_proj(attended.reshape(*u.shape[:-1],-1))
        mlps[number]=layer.mlp(layer.post_attention_layernorm(residual))
        h=residual+mlps[number]
        if number in pending:
            source,value=pending.pop(number)
            h=h+(adapter(m.extensions[str(source)],value)*scale(h)).to(h.dtype)
        history.append(h)
    hidden=m.backbone.model.norm(h)
    lm=m.lm_statistics(ctx,hidden)[0].sum()
    losses=[]
    for layer,pred in predictions.items():
        raw=sum(mlps[i].detach().float() for i in range(layer,layer+4))
        j=m.mean_index[layer];v=m.sigma2[j]
        target=((raw-m.mu[j])*torch.rsqrt(v.clamp_min(.01*v.median())+1e-6)).clamp(-10,10)
        losses.append(((1-F.cosine_similarity(pred.float(),target,dim=-1,eps=1e-6))*ctx.valid).sum())
    return hidden,lm,torch.stack(losses).mean()


class ProxyExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_parent_initialization_rng_and_budget(self):
        parent=model('P6-iso');expected=parent(batch())
        for arm in P6_EXTENSION_ARMS:
            before=torch.random.get_rng_state().clone();m=model(arm)
            torch.testing.assert_close(torch.random.get_rng_state(),before,rtol=0,atol=0)
            for name,value in parent.state_dict().items():
                torch.testing.assert_close(m.state_dict()[name],value,rtol=0,atol=0)
            out=m(batch())
            for key in ('lm_sum','aux_sum','statistics'):
                torch.testing.assert_close(out[key],expected[key],rtol=0,atol=0,msg=arm)
            budget=compute_budget(cfg(),m.settings,8)[arm]
            actual=sum(p.numel() for p in m.heads.parameters())+sum(p.numel() for p in m.extensions.parameters())
            self.assertEqual(actual,budget['extra_parameters'])
            self.assertTrue(m.settings.isolate_estimator)
            self.assertEqual(m.auxiliary_weight(250),.1)
            with self.assertRaises(ValueError):resolve_proxy_settings(arm,settings(isolate_estimator=False))
        for first,second in (P6_ADAPTER_ARMS,P6_DEPTH_ARMS,P6_DEEP_ARMS):
            a,b=model(first),model(second)
            for name,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[name],rtol=0,atol=0)

    def test_active_forward_and_gradients_against_independent_formula(self):
        for arm in P6_EXTENSION_ARMS:
            with self.subTest(arm=arm):
                m=model(arm);activate(m);ref=copy.deepcopy(m);ctx=batch()
                out=m(ctx,collect_target_statistics=True)
                hidden,lm,aux=reference(ref,ctx)
                torch.testing.assert_close(out['lm_sum'],lm,rtol=1e-6,atol=2e-6)
                torch.testing.assert_close(out['aux_sum'],aux,rtol=1e-6,atol=2e-6)
                torch.testing.assert_close(m.hidden_states(ctx)[0],hidden,rtol=1e-6,atol=2e-6)
                objective(m,out).backward();(lm/out['lm_count']+.1*aux/out['aux_count']).backward()
                for (name,p),(_,q) in zip(m.named_parameters(),ref.named_parameters()):
                    self.assertEqual(p.grad is None,q.grad is None,name)
                    if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=3e-5,atol=2e-6,msg=arm+'/'+name)
                self.assertFalse(out['center_sums'].requires_grad)

    def test_isolation_and_useful_initial_gradients(self):
        for arm in P6_EXTENSION_ARMS:
            m=model(arm)
            out=m(batch());objective(m,out).backward()
            for module in m.extensions.values():
                weight=module.weight if arm in P6_DEPTH_ARMS else module.up.weight
                self.assertGreater(float(weight.grad.abs().sum()),0,arm)
            activate(m)
            for auxiliary in (False,True):
                m.zero_grad(set_to_none=True);out=m(batch())
                out['aux_sum' if auxiliary else 'lm_sum'].backward()
                if auxiliary:
                    self.assertTrue(all(p.grad is None for p in m.backbone.parameters()),arm)
                    self.assertTrue(all(h.alpha.grad is None for h in m.heads.values()),arm)
                for name,p in m.named_parameters():
                    if name.startswith('heads.') and not name.endswith('.alpha'):
                        learns=auxiliary or arm=='P6-iso-deep'
                        self.assertEqual(p.grad is not None and bool(p.grad.abs().sum()),learns,arm+'/'+name)
                    if name.startswith('extensions.'):
                        learns=auxiliary if arm in P6_DEPTH_ARMS else not auxiliary
                        self.assertEqual(p.grad is not None and bool(p.grad.abs().sum()),learns,arm+'/'+name)

    def test_checkpoint_recompute_and_bfloat16(self):
        for arm in P6_EXTENSION_ARMS:
            for bf16 in (False,True):
                reference_model=model(arm);activate(reference_model)
                for m in (reference_model,model(arm,checkpoint=True),model(arm,aux_recompute=True)):
                    m.load_state_dict(reference_model.state_dict())
                    with torch.autocast('cpu',dtype=torch.bfloat16,enabled=bf16):
                        out=m(batch(),collect_target_statistics=True);loss=objective(m,out)
                    loss.backward()
                    if m is reference_model:expected=out
                    else:
                        for key in ('lm_sum','aux_sum','center_sums','center_squares'):
                            torch.testing.assert_close(out[key],expected[key],rtol=0,atol=0,msg=arm+'/'+key)
                        for (name,p),(_,q) in zip(m.named_parameters(),reference_model.named_parameters()):
                            self.assertEqual(p.grad is None,q.grad is None,name)
                            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=.01 if bf16 else 2e-5,atol=2e-6,msg=arm+'/'+name)

    def test_compiled_active_extensions(self):
        compile_fn=torch.compile
        for arm in P6_EXTENSION_ARMS:
            eager=model(arm);activate(eager)
            with patch('torch.compile',side_effect=lambda fn:compile_fn(fn,backend='aot_eager',fullgraph=True)):
                compiled=model(arm,compile_estimator=True)
            compiled.load_state_dict(eager.state_dict(),strict=True)
            for m in (eager,compiled):objective(m,m(batch())).backward()
            for (name,p),(_,q) in zip(eager.named_parameters(),compiled.named_parameters()):
                self.assertEqual(p.grad is None,q.grad is None,name)
                if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=2e-5,atol=2e-7,msg=arm+'/'+name)

    def test_nonzero_document_isolation_causality_and_ablation(self):
        for arm in P6_EXTENSION_ARMS:
            m=model(arm);activate(m);ctx=batch();hidden=m.hidden_states(ctx)[0]
            changed=copy.deepcopy(ctx);changed.input_ids[:,:3]=20
            torch.testing.assert_close(m.hidden_states(changed)[0][:,3:],hidden[:,3:],rtol=0,atol=0)
            changed=copy.deepcopy(ctx);changed.input_ids[:,6:]=20
            torch.testing.assert_close(m.hidden_states(changed)[0][:,:6],hidden[:,:6],rtol=0,atol=0)
            with m.without_proxy():
                torch.testing.assert_close(m.hidden_states(ctx)[0],model('A').hidden_states(ctx)[0],rtol=0,atol=0)
            self.assertFalse(m.gates_disabled)

    def test_active_fa4_layout_reference_at_twelve_layers(self):
        from tests.test_fa4_baseline import reference_kernel
        config=cfg();config.num_hidden_layers=12;config.layer_types=['full_attention']*12
        with patch('deep_kv.fa4.load_kernel',return_value=(reference_kernel,{'version':'CPU-test-double'})):
            for arm in P6_EXTENSION_ARMS:
                pair=[ProxyModel.from_scratch(config,arm,consumer=2,deep_target=12,
                    proxy_settings=settings(),attention_backend=backend,
                    checkpoint_layers=False,checkpoint_lm=False,checkpoint_aux=False,lm_chunk=3)
                    for backend in ('sdpa','fa4')]
                outputs=[]
                for m in pair:
                    activate(m);out=m(batch());outputs.append(out);objective(m,out).backward()
                for key in ('lm_sum','aux_sum'):torch.testing.assert_close(outputs[0][key],outputs[1][key],rtol=2e-5,atol=2e-6)
                for (name,p),(_,q) in zip(pair[0].named_parameters(),pair[1].named_parameters()):
                    self.assertEqual(p.grad is None,q.grad is None,name)
                    if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=3e-4,atol=3e-6,msg=arm+'/'+name)

    def test_trainer_resume_reload_metadata_report_and_queue(self):
        from safetensors.torch import load_file
        from tests.test_proxy_training import proxy_fixture
        from tests.test_train import invoke
        from eval.models import load_checkpoint
        from deep_kv.report import report
        from scripts.proxy_12layer_queue import make,ARMS
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);recipe=proxy_fixture(root)
            recipe.update(checkpoint_layers=False,checkpoint_lm=False,checkpoint_aux=False,proxy_sync_optimization=True)
            for arm in ('A','P6-iso')+P6_EXTENSION_ARMS:
                args={**recipe,'arm':arm,'output_dir':str(root/arm)}
                invoke(root,{**args,'stop_after':2});invoke(root,args)
                loaded,_,_=load_checkpoint(root/arm,device='cpu',attention_backend='sdpa')
                self.assertEqual(loaded.wrapped.arm,arm)
                if arm in P6_EXTENSION_ARMS:
                    config=json.loads((root/arm/'train_config.json').read_text())
                    self.assertEqual(config['proxy_target']['extension'],anticipatory_target_metadata(arm)['extension'])
                    invoke(root,{**args,'output_dir':str(root/('full-'+arm))})
                    resumed=load_file(root/arm/'model.safetensors');full=load_file(root/('full-'+arm)/'model.safetensors')
                    for k in resumed:torch.testing.assert_close(resumed[k],full[k],rtol=0,atol=0,msg=arm+'/'+k)
                    with self.assertRaisesRegex(ValueError,'Resume configuration'):
                        invoke(root,{**args,'arm':'P6-iso'})
            result=report(root,('A','P6-iso')+P6_EXTENSION_ARMS)
            for a,b in (P6_ADAPTER_ARMS,P6_DEPTH_ARMS,P6_DEEP_ARMS):self.assertIn(a+'-'+b,result['nll_differences'])
            path=root/P6_EXTENSION_ARMS[0]/'train_config.json';bad=json.loads(path.read_text())
            bad['proxy_target']['extension']['source']='shallow_norm';path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,'target metadata'):report(root,('A',P6_EXTENSION_ARMS[0]))
            jobs=make('proxy_heads.12layer.b200.json',root/'queue.json',extensions=True)
            gpu=[j for j in jobs if 'gpus' in j]
            self.assertEqual(len(gpu),len(ARMS)+len(P6_EXTENSION_ARMS))
            self.assertTrue(all(j['gpus']==list(range(8)) for j in gpu))
            self.assertEqual([j['argv'][j['argv'].index('--arm')+1] for j in gpu],list(ARMS+P6_EXTENSION_ARMS))

    def test_two_rank_zero_auxiliary_weight_then_active(self):
        from tests.test_proxy_training import proxy_fixture
        for arm in P6_EXTENSION_ARMS:
            with self.subTest(arm=arm),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);recipe=proxy_fixture(root,world=2)
                recipe.update(arm=arm,ddp_backend='gloo',ddp_find_unused_parameters=False,
                    checkpoint_layers=False,checkpoint_lm=False,checkpoint_aux=False,
                    proxy_sync_optimization=True,logging_steps=2,eval_on_start=False)
                path=root/'ddp.json';path.write_text(json.dumps(recipe));log=root/'ddp.log'
                with log.open('w') as output:
                    result=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone',
                        '--nproc_per_node=2','train.py',str(path)],cwd=Path(__file__).resolve().parents[1],
                        env={**os.environ,'OMP_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''},
                        stdout=output,stderr=subprocess.STDOUT,timeout=180)
                self.assertEqual(result.returncode,0,log.read_text()[-12000:])
                value=json.loads((Path(recipe['output_dir'])/'result.json').read_text())
                self.assertEqual(value['global_step'],3)
                self.assertTrue(torch.isfinite(torch.tensor(value['evaluation']['eval_aux_loss'])))


if __name__=='__main__':unittest.main()
