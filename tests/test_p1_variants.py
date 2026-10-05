"""P1 values-only routing and shorter targets at fixed injection layers."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file
from transformers import HfArgumentParser
import train
from deep_kv.__main__ import jobs
from deep_kv.proxy import ProxyModel, compute_budget, proxy_layers
from tests.test_proxy_heads import model, batch, cfg, settings, objective
from tests.test_proxy_training import proxy_fixture
from tests.test_train import invoke
from tests.test_fa4_baseline import reference_kernel

CASES = [dict(kv_mode='v',lookahead=4),dict(kv_mode='kv',lookahead=2),dict(kv_mode='kv',lookahead=3)]


class P1VariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_values_only_uses_native_queries_and_keys(self):
        m=model('P1-block',kv_mode='v',alpha_init=1.)
        layer=m.backbone.model.layers[1];ctx=batch();u=torch.randn(1,8,32,requires_grad=True)
        z=(u+torch.randn_like(u)).detach().requires_grad_()
        rotary=m.backbone.model.rotary_emb(u,ctx.position_ids);captured=[]
        def attention(module,q,k,v,mask,**kwargs):
            captured.append((q,k,v))
            return v.repeat_interleave(module.num_key_value_groups,dim=1).transpose(1,2),None
        with patch('deep_kv.proxy.ALL_ATTENTION_FUNCTIONS.get_interface',return_value=attention):
            m.attention(layer,u,None,ctx.allowed(),rotary)
            m.attention(layer,u,z,ctx.allowed(),rotary)
        native,proxy=captured
        for i in (0,1):torch.testing.assert_close(native[i],proxy[i],rtol=0,atol=0)
        torch.testing.assert_close(native[2][:,:1],proxy[2][:,:1],rtol=0,atol=0)
        expected=layer.self_attn.v_proj(z).view(1,8,2,layer.self_attn.head_dim).transpose(1,2)
        torch.testing.assert_close(proxy[2][:,1:],expected[:,1:],rtol=0,atol=0)
        self.assertGreater((native[2][:,1:]-proxy[2][:,1:]).abs().max(),0)
        self.assertIsNone(torch.autograd.grad(proxy[0].sum()+proxy[1].sum(),z,allow_unused=True)[0])
        self.assertGreater(torch.autograd.grad(proxy[2].square().sum(),z)[0].abs().sum(),0)

    def test_short_windows_use_exact_mlp_sum_and_fixed_placement(self):
        full_cfg=cfg();full_cfg.num_hidden_layers=28
        for k in (2,3):
            self.assertEqual(proxy_layers(full_cfg,'P1',k,list(range(2,25,2))),tuple(range(2,25,2)))
        baseline=model('P1-block',alpha_init=1.)
        for k in (2,3):
            m=model('P1-block',lookahead=k,layers=[2,4],alpha_init=1.)
            self.assertEqual(m.layers,baseline.layers)
            for name,value in m.state_dict().items():torch.testing.assert_close(value,baseline.state_dict()[name],rtol=0,atol=0)
            self.assertEqual(compute_budget(cfg(),settings(lookahead=k,layers=[2,4]),8),
                             compute_budget(cfg(),settings(),8))
            mlps={};targets={};handles=[]
            for i,layer in enumerate(m.backbone.model.layers,1):
                def capture(module,args,out,index=i):mlps[index]=out.detach().clone()
                handles.append(layer.mlp.register_forward_hook(capture))
            original=m.normalize_target
            def target(value,key,**kwargs):
                targets[key]=value.detach().clone();self.assertFalse(value.requires_grad)
                return original(value,key,**kwargs)
            try:
                with patch.object(m,'normalize_target',side_effect=target):out=m(batch())
            finally:
                for h in handles:h.remove()
            self.assertEqual(set(targets),{2,4})
            for layer,value in targets.items():
                expected=sum(mlps[i].float() for i in range(layer,layer+k))
                torch.testing.assert_close(value,expected,rtol=0,atol=0)
            torch.testing.assert_close(out['lm_sum'],baseline(batch())['lm_sum'],rtol=0,atol=0)
        for layers in ([],[4,2],[2,2],[1],[8]):
            with self.assertRaises(ValueError):model('P1-block',lookahead=2,layers=layers)
        with self.assertRaises(ValueError):model('P3-block',kv_mode='v')
        with self.assertRaises(ValueError):model('P3-block',layers=[2,4])

    def test_document_isolation_zero_gate_and_block_aux_routing(self):
        for case in CASES:
            m=model('P1-block',**case,layers=[2,4],alpha_init=1.);ctx=batch()
            captured=[]
            h=m.backbone.model.embed_tokens.register_forward_hook(lambda mod,args,out:captured.append(out))
            try:
                hidden=m.hidden_states(ctx)[0]
                grad=torch.autograd.grad(hidden[:,3:].square().sum(),captured[-1])[0]
                self.assertEqual(grad[:,:3].abs().max(),0)
                changed=copy.deepcopy(ctx);changed.input_ids[:,:3]=20
                torch.testing.assert_close(hidden[:,3:],m.hidden_states(changed)[0][:,3:],rtol=0,atol=0)
            finally:h.remove()
            out=m(ctx);out['aux_sum'].backward()
            self.assertFalse(any(p.grad is not None for p in m.backbone.parameters()))
            zero=model('P1-block',**case,layers=[2,4])
            torch.testing.assert_close(zero(ctx)['lm_sum'],model('A')(ctx)['lm_sum'],rtol=0,atol=1e-5)

    def test_sdpa_and_varlen_reference_gradients_with_checkpointing(self):
        with patch('deep_kv.fa4.load_kernel',return_value=(reference_kernel,{'version':'CPU-test-double'})):
            for case in CASES:
                for checkpoint in (False,True):
                    options=settings(**case,layers=[2,4],alpha_init=1.)
                    models=[ProxyModel.from_scratch(cfg(),'P1-block',consumer=2,deep_target=8,
                        proxy_settings=options,attention_backend=backend,checkpoint_layers=checkpoint,
                        checkpoint_lm=checkpoint,checkpoint_aux=checkpoint,lm_chunk=3) for backend in ('sdpa','fa4')]
                    outputs=[]
                    for m in models:
                        out=m(batch(),collect_target_statistics=True);objective(m,out).backward();outputs.append(out)
                    for name in ('lm_sum','aux_sum','center_sums','center_squares','center_counts'):
                        torch.testing.assert_close(outputs[0][name],outputs[1][name],rtol=2e-4,atol=2e-5)
                    for (name,p),(_,q) in zip(models[0].named_parameters(),models[1].named_parameters()):
                        self.assertEqual(p.grad is None,q.grad is None,name)
                        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=5e-4,atol=3e-6,msg=name)
                    for m in models:
                        for p in m.heads.parameters():self.assertGreater(p.grad.abs().sum(),0)

    def test_real_trainer_resume_and_queue_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);recipe=proxy_fixture(root)
            for index,case in enumerate(CASES):
                args={**recipe,'arm':'P1-block','proxy_alpha_init':1.,'proxy_layers':[2,4],
                      **{'proxy_'+k:v for k,v in case.items()},'output_dir':str(root/str(index))}
                invoke(root,{**args,'stop_after':2})
                for changed in ({'proxy_kv_mode':'kv' if case['kv_mode']=='v' else 'v'},
                                {'proxy_lookahead':1},{'proxy_layers':[2]}):
                    with self.assertRaisesRegex(ValueError,'Resume configuration'):invoke(root,{**args,**changed})
                invoke(root,args);invoke(root,{**args,'output_dir':str(root/f'full-{index}')})
                resumed=load_file(root/str(index)/'model.safetensors');full=load_file(root/f'full-{index}'/'model.safetensors')
                for key in full:torch.testing.assert_close(resumed[key],full[key],rtol=0,atol=0)
                saved=json.loads((root/str(index)/'train_config.json').read_text())
                self.assertEqual(saved['pilot']['proxy_layers'],[2,4])
                self.assertEqual(saved['pilot'].get('proxy_kv_mode','kv'),case['kv_mode'])
                path=root/'recipe.json';path.write_text(json.dumps(args))
                job=jobs(path,arms=['P1-block'],seeds=[42])['jobs'][0]
                parser=HfArgumentParser((train.ModelArguments,train.DataArguments,train.PilotArguments,train.TrainingArguments))
                cli=job['argv'][job['argv'].index(str(Path(train.__file__).resolve()))+1:]
                _,_,pilot,_=parser.parse_args_into_dataclasses(cli)
                self.assertEqual(pilot.proxy_layers,[2,4]);self.assertEqual(pilot.proxy_lookahead,case['lookahead'])
                self.assertEqual(pilot.proxy_kv_mode,case['kv_mode'])
                from scripts.evaluate_proxy_gates import run_sweep
                run_sweep(root/str(index),root/f'gate-eval-{index}',step=3)
