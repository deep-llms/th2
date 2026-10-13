"""CPU-only gradient and queue contracts; no real-data or GPU training."""
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

from deep_kv import P6_ROUTING_ARMS, auxiliary_routes, anticipatory_layout
from deep_kv.proxy import ProxyModel, ProxySettings, resolve_proxy_settings
from scripts.proxy_12layer_queue import ARMS, make
from tests.test_proxy_heads import model, batch, cfg, settings


class AuxiliaryRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_auxiliary_gradients_match_independent_graph(self):
        for arm in ('P6-iso',)+P6_ROUTING_ARMS:
            with self.subTest(arm=arm):
                actual, reference = model(arm), model('P6-iso')
                reference.load_state_dict(actual.state_dict(), strict=True)
                routes=auxiliary_routes(arm)
                ctx=batch(); raw={}; sources={}; handles=[]
                for i, block in enumerate(reference.backbone.model.layers,1):
                    handles.append(block.mlp.register_forward_hook(
                        lambda m,a,o,i=i:raw.__setitem__(i,o)))
                    if i in reference.layers:
                        # First input-layernorm output is the predictor source;
                        # P6 calls the same norm again after residual injection.
                        def capture(m,a,o,i=i):
                            if i not in sources:sources[i]=o
                        handles.append(block.input_layernorm.register_forward_hook(capture))
                expected_forward=reference(ctx,compute_auxiliary_losses=False)
                losses=[]
                for layer in reference.layers:
                    source=sources[layer] if routes['shallow'] else sources[layer].detach()
                    pred=reference.heads[str(layer)].estimates(source)[0]
                    target=sum(raw[i].float() for i in range(layer,layer+4))
                    if not routes['target']:target=target.detach()
                    j=reference.mean_index[layer]
                    variance=reference.sigma2[j]
                    inv=torch.rsqrt(variance.clamp_min(.01*variance.median())+1e-6)
                    target=((target-reference.mu[j])*inv).clamp(-10,10)
                    losses.append(((1-F.cosine_similarity(pred.float(),target,dim=-1,eps=1e-6))*ctx.valid).sum())
                expected=torch.stack(losses).mean()
                out=actual(ctx,collect_target_statistics=True)
                torch.testing.assert_close(out['lm_sum'],expected_forward['lm_sum'],rtol=0,atol=0)
                torch.testing.assert_close(out['aux_sum'],expected,rtol=1e-6,atol=2e-6)
                out['aux_sum'].backward();expected.backward()
                for (name,p),(_,q) in zip(actual.named_parameters(),reference.named_parameters()):
                    self.assertEqual(p.grad is None,q.grad is None,name)
                    if p.grad is not None:
                        self.assertTrue(torch.isfinite(p.grad).all(),name)
                        torch.testing.assert_close(p.grad,q.grad,rtol=2e-5,atol=2e-6,msg=arm+'/'+name)
                backbone_grad=sum(float(p.grad.abs().sum()) for p in actual.backbone.parameters() if p.grad is not None)
                self.assertEqual(backbone_grad>0,any(routes.values()))
                for key in ('center_sums','center_squares','center_counts','clip_counts'):
                    self.assertFalse(out[key].requires_grad,key)
                for h in handles:h.remove()

    def test_forward_and_lm_gradient_are_unchanged(self):
        ref=model('P6-iso');ctx=batch();expected=ref(ctx);expected['lm_sum'].backward()
        for arm in P6_ROUTING_ARMS:
            m=model(arm);out=m(ctx)
            for key in ('lm_sum','aux_sum','statistics'):
                torch.testing.assert_close(out[key],expected[key],rtol=0,atol=0)
            out['lm_sum'].backward()
            for (name,p),(_,q) in zip(m.named_parameters(),ref.named_parameters()):
                self.assertEqual(p.grad is None,q.grad is None,name)
                if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
                if name.startswith('heads.') and not name.endswith('.alpha'):self.assertIsNone(p.grad)
            m.zero_grad(set_to_none=True)
            out=m(ctx,auxiliary_grad=False,collect_target_statistics=True)
            self.assertFalse(out['aux_sum'].requires_grad)

    def test_target_normalization_clipping_and_stopped_statistics(self):
        m=model('P6-iso-targetgrad')
        x=torch.tensor([-20.,-.2,.4,20.]).repeat(8).reshape(1,1,32).requires_grad_()
        y=m.normalize_target(x,2,target_grad=True)
        y.sum().backward()
        self.assertEqual(int((x.grad==0).sum()),16)
        torch.testing.assert_close(x.grad[0,0,1:3],torch.ones(2)/(1+1e-6)**.5)
        self.assertFalse(m.normalize_target(x,2).requires_grad)
        self.assertFalse(m.mu.requires_grad);self.assertFalse(m.sigma2.requires_grad)

    def test_twelve_layer_varlen_attention_reference(self):
        # Exercise the FA4 adapter/layout on CPU; this is not a CUDA-kernel test.
        from tests.test_fa4_baseline import reference_kernel
        config=cfg();config.num_hidden_layers=12;config.layer_types=['full_attention']*12
        with patch('deep_kv.fa4.load_kernel',return_value=(reference_kernel,{'version':'CPU-test-double'})):
            for arm in P6_ROUTING_ARMS:
                pair=[ProxyModel.from_scratch(config,arm,consumer=2,deep_target=12,
                    proxy_settings=settings(),attention_backend=backend,
                    checkpoint_layers=False,checkpoint_lm=False,checkpoint_aux=False,lm_chunk=3)
                    for backend in ('sdpa','fa4')]
                outputs=[]
                for m in pair:
                    out=m(batch(),collect_target_statistics=True);outputs.append(out)
                    (out['lm_sum']/out['lm_count']+.1*out['aux_sum']/out['aux_count']).backward()
                for key in ('lm_sum','aux_sum','center_sums','center_squares'):
                    torch.testing.assert_close(outputs[0][key],outputs[1][key],rtol=2e-5,atol=2e-6)
                for (name,p),(_,q) in zip(pair[0].named_parameters(),pair[1].named_parameters()):
                    self.assertEqual(p.grad is None,q.grad is None,name)
                    if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=3e-4,atol=2e-6,msg=name)

    def test_all_nine_arms_at_twelve_layers(self):
        config=cfg();config.num_hidden_layers=12;config.layer_types=['full_attention']*12
        config.num_attention_heads=8;config.num_key_value_heads=4
        shared=None
        for arm in ARMS:
            m=ProxyModel.from_scratch(config,arm,consumer=2,deep_target=12,
                proxy_settings=ProxySettings(groups=2,width=8,features=9,chunk_size=2),
                checkpoint_layers=False,checkpoint_lm=False,checkpoint_aux=False,lm_chunk=3)
            backbone={k:v.clone() for k,v in m.backbone.state_dict().items()}
            if shared is None:shared=backbone
            for k,v in backbone.items():torch.testing.assert_close(v,shared[k],rtol=0,atol=0)
            out=m(batch(),collect_target_statistics=True)
            loss=out['lm_sum']/out['lm_count']+m.auxiliary_weight(250)*out['aux_sum']/out['aux_count'].clamp_min(1)
            loss.backward()
            self.assertTrue(torch.isfinite(loss),arm)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None),arm)
            self.assertEqual(m.layers,() if arm=='A' else (2,6) if arm=='P6-iso-sparse' else (2,4,6,8))

    def test_lowaux_only_changes_weight_and_keeps_lm_route(self):
        a,b=model('P6'),model('P6-lowaux')
        for name,value in a.state_dict().items():
            torch.testing.assert_close(value,b.state_dict()[name],rtol=0,atol=0)
        for step in (0,1,125,250,1000):
            self.assertAlmostEqual(b.auxiliary_weight(step),.3*a.auxiliary_weight(step))
        self.assertFalse(b.settings.isolate_estimator)
        b(batch())['lm_sum'].backward()
        self.assertGreater(float(b.heads['2'].w1.weight.grad.abs().sum()),0)
        with self.assertRaises(ValueError):resolve_proxy_settings('P6-lowaux',settings(lambda_max=.1))

    def test_twelve_layer_depth_and_queue(self):
        from train import configure_depth,ModelArguments,DataArguments,PilotArguments
        from transformers import HfArgumentParser,TrainingArguments
        config=cfg();original=copy.deepcopy(config.to_dict());configure_depth(config,12)
        self.assertEqual(config.num_hidden_layers,12)
        self.assertEqual(config.layer_types,['full_attention']*12)
        self.assertEqual(config.hidden_size,original['hidden_size'])
        for depth in (0,1,True,-2):
            with self.assertRaises(ValueError):configure_depth(config,depth)
        self.assertEqual(anticipatory_layout('P6-iso',12)[0],(2,4,6,8))
        self.assertEqual(anticipatory_layout('P6-iso-sparse',12)[0],(2,6))
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'jobs.json'
            items=make('proxy_heads.12layer.b200.json',output)
            self.assertEqual(len(items),20)
            parser=HfArgumentParser((ModelArguments,DataArguments,PilotArguments,TrainingArguments))
            for job,arm in zip([j for j in items if 'gpus' in j],ARMS):
                self.assertEqual(job['gpus'],list(range(8)))
                argv=job['argv'];start=next(i for i,v in enumerate(argv) if v.endswith('/train.py'))+1
                ma,da,pa,ta=parser.parse_args_into_dataclasses([*argv[start:],'--use_cpu','true'])
                resolved=resolve_proxy_settings(arm,ProxySettings(**{
                    k:getattr(pa,'proxy_'+k) for k in ProxySettings.__dataclass_fields__}))
                self.assertEqual(resolved.lambda_max,.03 if arm=='P6-lowaux' else .1)
                self.assertEqual((ma.num_hidden_layers,ta.max_steps,ta.warmup_steps),(12,2500,125))
                self.assertEqual(pa.arm,arm);self.assertEqual(ta.save_total_limit,0)
                self.assertTrue(pa.proxy_sync_optimization)
                self.assertEqual(job['required_outputs'][0]['json_equals']['status'],'complete')
                validator=items[items.index(job)+1]
                self.assertIn('--num-hidden-layers',validator['argv'])
            self.assertFalse(any('finetune' in str(j) or 'eval.suite' in str(j) for j in items))
            with self.assertRaises(FileExistsError):make('proxy_heads.12layer.b200.json',output)

    def test_trainer_resume_reload_and_routing_metadata(self):
        from safetensors.torch import load_file
        from tests.test_proxy_training import proxy_fixture
        from tests.test_train import invoke
        from eval.models import load_checkpoint
        from deep_kv.report import report
        arms=('A','P6-iso',*P6_ROUTING_ARMS,'P6','P6-lowaux')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);recipe=proxy_fixture(root)
            recipe.update(num_hidden_layers=12,deep_target=12,proxy_sync_optimization=True,
                          checkpoint_layers=False,checkpoint_lm=False,checkpoint_aux=False)
            for arm in arms:
                args={**recipe,'arm':arm,'output_dir':str(root/arm)}
                invoke(root,{**args,'stop_after':2})
                invoke(root,args)
                loaded,_,_=load_checkpoint(root/arm,device='cpu',attention_backend='sdpa')
                self.assertEqual(len(loaded.wrapped.backbone.model.layers),12)
                self.assertEqual(loaded.wrapped.arm,arm)
                if arm in P6_ROUTING_ARMS:
                    config_path=root/arm/'train_config.json'
                    config=json.loads(config_path.read_text())
                    self.assertEqual(config['proxy_target']['auxiliary_routes'],auxiliary_routes(arm))
                    invoke(root,{**args,'output_dir':str(root/('full-'+arm))})
                    resumed=load_file(root/arm/'model.safetensors')
                    full=load_file(root/('full-'+arm)/'model.safetensors')
                    for k in resumed:torch.testing.assert_close(resumed[k],full[k],rtol=0,atol=0,msg=arm+'/'+k)
                    with self.assertRaisesRegex(ValueError,'Resume configuration'):
                        invoke(root,{**args,'arm':'P6-iso'})
            summary=report(root,arms)
            self.assertIn('P6-lowaux-P6',summary['nll_differences'])
            for arm in P6_ROUTING_ARMS:self.assertIn(arm+'-P6-iso',summary['nll_differences'])
            # An edited arm label cannot disguise a different target route.
            path=root/'P6-iso-targetgrad'/'train_config.json'
            bad=json.loads(path.read_text());bad['proxy_target']['auxiliary_routes']['target']=False
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,'target metadata'):report(root,arms)

    def test_two_rank_zero_weight_then_joint_backbone_routes(self):
        from tests.test_proxy_training import proxy_fixture
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);recipe=proxy_fixture(root,world=2)
            recipe.update(arm='P6-iso-bothgrad',num_hidden_layers=12,deep_target=12,
                          ddp_backend='gloo',ddp_find_unused_parameters=False,
                          checkpoint_layers=False,checkpoint_lm=False,checkpoint_aux=False,
                          proxy_sync_optimization=True,logging_steps=2,eval_on_start=False)
            path=root/'ddp.json';path.write_text(json.dumps(recipe))
            log=root/'ddp.log'
            with log.open('w') as out:
                result=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone',
                    '--nproc_per_node=2','train.py',str(path)],cwd=Path(__file__).resolve().parents[1],
                    env={**os.environ,'OMP_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''},
                    stdout=out,stderr=subprocess.STDOUT,timeout=240)
            self.assertEqual(result.returncode,0,log.read_text()[-16000:])
            value=json.loads((Path(recipe['output_dir'])/'result.json').read_text())
            self.assertEqual(value['global_step'],3)
            self.assertTrue(torch.isfinite(torch.tensor(value['evaluation']['eval_aux_loss'])))


if __name__=='__main__':unittest.main()
