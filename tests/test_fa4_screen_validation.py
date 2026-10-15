"""CPU checks for the production-step checkpoint validator, without GPU work."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from safetensors.torch import save_file
from scripts.check_fa4_proxy import validate
from deep_kv.p7_variants import TUNING_ARMS, NOAUX_ARMS, KEYONLY_ARMS, VALUEONLY_ARMS, settings_overrides
from deep_kv import MEMORY_ARMS, SIMPLE_MEMORY_ARMS, P7_EXPLORATION_ARMS
from deep_kv import P6_EXTENSION_ARMS, P6_DEPTH_ARMS, anticipatory_target_metadata
from deep_kv import FOLLOWUP_ARMS, DELAYED_ARMS, P6_CONSUMER_ARMS, P6_SPARSE_DEEP, followup_metadata, P7_DEEP_ARMS


class ScreenValidationTests(unittest.TestCase):
    def test_smoke_and_screen_cutoffs_and_missing_rank_state(self):
        for backend,variant,steps,interval in (('fa4','P3-block',3,1),('fa4','P3-block',2500,10),
                                             ('fa4','P6-iso-sparse',25,10),('fa4','P7-simple-short',25,10),
                                             ('fa4','P4-4h',25,1),('fa4','P4-iso-4h',2500,10),
                                             ('sdpa','P4-4h',3,1),('sdpa','P4-iso-4h',2500,10),
                                             ('fa4','P7',25,1),('fa4','P7-kq',25,1),('sdpa','P7-ems',25,1),
                                             ('fa4','P7-simple',25,1),('sdpa','P7-simple',25,1),
                                             ('fa4','P7-simple-sparse',25,1),('sdpa','P7-simple-sparse',25,1),
                                             ('fa4','P7-simple-short',25,1),('sdpa','P7-simple-short',25,1)) + tuple(
                (backend,arm,25,1) for arm in P6_EXTENSION_ARMS+FOLLOWUP_ARMS+P7_DEEP_ARMS+P7_EXPLORATION_ARMS+TUNING_ARMS for backend in ('sdpa','fa4')):
            with self.subTest(steps=steps,arm=variant),tempfile.TemporaryDirectory() as tmp:
                layers=(2,6,10,14,18,22) if '-sparse' in variant else tuple(range(2,25,2))
                root=Path(tmp);arm=root/f'seed-42/{variant}';checkpoint=arm/f'checkpoint-{steps}'
                checkpoint.mkdir(parents=True)
                config=dict(pilot=dict(attention_backend=backend,checkpoint_layers=False,checkpoint_lm=False,checkpoint_aux=False),
                    data=dict(isolate_documents=True,eval_rows=32),world_size=8,
                    training=dict(per_device_train_batch_size=16,gradient_accumulation_steps=4,warmup_steps=1430,seed=42,data_seed=42,logging_steps=interval))
                state=dict(global_step=steps,max_steps=28600,log_history=[dict(step=i,loss=3.,grad_norm=.2) for i in range(interval,steps+1,interval)])
                result=dict(arm=variant,global_step=steps,status='stopped',schedule_steps=28600,input_tokens=steps*1048576,
                    attention_runtime=dict(version='4.0.0b33'),sdpa_receipts=[],fa4_receipts=['train.json','eval.json'],
                    evaluation=dict(eval_lm_loss=3.,eval_rows=32),training_cost=dict(peak_cuda_allocated_bytes=128*2**30))
                if backend=='sdpa':
                    result.update(sdpa_receipts=['train.json','eval.json'],fa4_receipts=[],attention_runtime={})
                if variant in MEMORY_ARMS:
                    config['pilot']['proxy_target_version']='p4p6-r1' if variant in SIMPLE_MEMORY_ARMS else 'p7-r1'
                    result['evaluation'].update(eval_cos_loss=1.,eval_aux_loss=1.)
                    if variant not in SIMPLE_MEMORY_ARMS:result['evaluation'].update(eval_rel_loss=.5,eval_aux_loss=1.25)
                    result['evaluation'].update({f'eval_proxy_layer_{layer}_attention_mass':.5 for layer in layers})
                if variant in P6_EXTENSION_ARMS+P6_SPARSE_DEEP:
                    config['proxy_target']=anticipatory_target_metadata(variant)
                if variant in FOLLOWUP_ARMS+P7_DEEP_ARMS+P7_EXPLORATION_ARMS+TUNING_ARMS:config.setdefault('proxy_target',{}).update(followup_metadata(variant))
                if variant in NOAUX_ARMS:
                    config['pilot']['proxy_lambda_max']=0.;result['evaluation']['eval_aux_loss']=0.
                if variant=='P7-simple-lm-nexttoken':
                    result['evaluation'].update(eval_next_token_loss=1.,eval_next_token_targets=8)
                if variant in TUNING_ARMS:
                    config['pilot'].update(settings_overrides(variant))
                    config['model_config']=dict(num_hidden_layers=28,num_attention_heads=16,num_key_value_heads=8,head_dim=8,hidden_size=32)
                for name,value in (('train_config',config),('trainer_state',state),('result',result)):
                    (arm/f'{name}.json').write_text(json.dumps(value))
                (checkpoint/'trainer_state.json').write_text(json.dumps(state))
                for phase in ('train','eval'):
                    calls=28+len(layers) if variant in MEMORY_ARMS and variant not in DELAYED_ARMS else 28
                    if variant in VALUEONLY_ARMS:calls=28
                    gradients=calls
                    if variant in P6_CONSUMER_ARMS:
                        calls+=2*len(layers);gradients+=len(layers)
                    receipt=dict(arm=variant,phase=phase,calls=calls,world_size=8,kernel_dtype='torch.bfloat16',
                        positions='reset_per_document',document_isolation=True,causal=True,dense_mask=False,
                        query_gradient_calls=gradients,key_gradient_calls=gradients,value_gradient_calls=gradients,global_step=0)
                    if backend=='sdpa':
                        receipt.update(calls=[dict(mask=dict(dtype='torch.bool'),is_causal=False,
                            backward_operators=['fused_backward'] if i<gradients else [],
                            **{k:dict(requires_grad=i<gradients) for k in ('query','key','value')}) for i in range(calls)],
                            has_math=False,unattributed_calls=[],unattributed_backward=[],
                            checkpointing=dict(checkpoint_layers=False,checkpoint_lm=False,checkpoint_aux=False))
                    (arm/f'{phase}.json').write_text(json.dumps(receipt))
                for name in ['optimizer.pt','scheduler.pt',*[f'rng_state_{i}.pth' for i in range(8)]]:
                    (checkpoint/name).write_bytes(b'fixture')
                heads=({f'heads.{layer}.{suffix}':torch.ones(3) for layer in layers
                        for suffix in (('w1.weight','w2.weight') if variant in SIMPLE_MEMORY_ARMS else ('conv','w_in.weight','w_out.weight'))+
                        (() if variant in VALUEONLY_ARMS else ('k_proj.weight','k_norm.weight'))+
                        (() if variant in ('P7-kq',)+KEYONLY_ARMS else ('v_proj.weight',))} if variant in MEMORY_ARMS else
                       {'heads.2.alpha':torch.ones(1,3)*.1})
                if variant in TUNING_ARMS:
                    g=settings_overrides(variant)['proxy_groups']
                    for layer in layers:
                        for suffix in ('k_proj.weight','v_proj.weight','k_norm.weight'):
                            name=f'heads.{layer}.{suffix}'
                            if name in heads:heads[name]=torch.ones(8) if suffix=='k_norm.weight' else torch.ones(g*8,32)
                        if variant.endswith('-nonlinear'):
                            for kind in ('key_transform','value_transform'):
                                for index in (0,2):heads[f'heads.{layer}.{kind}.{index}.weight']=torch.ones(32,32)
                        if variant.endswith('-gated'):heads[f'heads.{layer}.value_gate']=torch.ones(2*g)
                if variant in P6_EXTENSION_ARMS+P6_SPARSE_DEEP+P7_DEEP_ARMS:
                    heads.update({f'extensions.{layer}.{suffix}':torch.ones(3) for layer in layers
                        for suffix in (('weight',) if variant in P6_DEPTH_ARMS else ('down.weight','up.weight'))})
                if variant in DELAYED_ARMS:heads['proxy_updates']=torch.tensor(steps,dtype=torch.long)
                save_file(dict(mu_initialized=torch.tensor(True),mu=torch.ones(2,3),sigma2=torch.ones(2,3),
                    **heads),checkpoint/'model.safetensors')
                args=SimpleNamespace(run_dir=str(root),output=str(root/'passed.json'),steps=steps,seed=42,arms=[variant],attention_backend=backend)
                validate(args);self.assertEqual(json.loads((root/'passed.json').read_text())['steps'],steps)
                config['pilot']['proxy_lambda_max']=.3
                (arm/'train_config.json').write_text(json.dumps(config))
                args.expected_lambda=.1;args.output=str(root/'wrong-lambda.json')
                with self.assertRaises(AssertionError):validate(args)
                args.expected_lambda=.3;args.output=str(root/'right-lambda.json')
                if variant in NOAUX_ARMS:
                    with self.assertRaises(AssertionError):validate(args)
                    config['pilot']['proxy_lambda_max']=0.
                    (arm/'train_config.json').write_text(json.dumps(config))
                else:validate(args)
                args.expected_lambda=None
                if variant in DELAYED_ARMS:
                    # A long delayed run must contain post-introduction receipts.
                    # Correct final weights and only dormant audits are insufficient.
                    late_steps=501
                    late_state={**state,'global_step':late_steps,'log_history':[
                        dict(step=i,loss=3.,grad_norm=.2) for i in range(1,late_steps+1)]}
                    late_result={**result,'global_step':late_steps,'input_tokens':late_steps*1048576}
                    for name,value in [('trainer_state',late_state),('result',late_result)]:
                        (arm/f'{name}.json').write_text(json.dumps(value))
                    args.steps=late_steps;args.output=str(root/'missing-active-audit.json')
                    with self.assertRaises(AssertionError):validate(args)
                    self.assertFalse(Path(args.output).exists())
                    (arm/'trainer_state.json').write_text(json.dumps(state))
                    (arm/'result.json').write_text(json.dumps(result));args.steps=steps
                # Disposable smoke schedules must be explicit; production defaults
                # still reject artifacts with an unexpected schedule or warmup.
                args.output=str(root/'wrong-schedule.json');args.schedule_steps=28601
                with self.assertRaises(AssertionError):validate(args)
                args.schedule_steps=28600;args.warmup_steps=1431
                with self.assertRaises(AssertionError):validate(args)
                args.warmup_steps=1430
                if steps < 100:
                    smoke_config={**config,'training':{**config['training'],'warmup_steps':5}}
                    smoke_state={**state,'max_steps':100}
                    smoke_result={**result,'schedule_steps':100}
                    for name,value in [('train_config',smoke_config),('trainer_state',smoke_state),('result',smoke_result)]:
                        (arm/f'{name}.json').write_text(json.dumps(value))
                    args.output=str(root/'smoke.json');args.schedule_steps=100;args.warmup_steps=5
                    validate(args)
                    for name,value in [('train_config',config),('trainer_state',state),('result',result)]:
                        (arm/f'{name}.json').write_text(json.dumps(value))
                    args.schedule_steps=28600;args.warmup_steps=1430
                if variant=='P6-iso-sparse' and backend=='fa4':
                    # A completed 12-layer schedule, rather than a forced cutoff.
                    small_config={**config,'model_config':{'num_hidden_layers':12},
                                  'training':{**config['training'],'warmup_steps':1}}
                    small_state={**state,'max_steps':steps}
                    small_result={**result,'schedule_steps':steps,'status':'complete'}
                    saved_receipts={}
                    for name,value in [('train_config',small_config),('trainer_state',small_state),('result',small_result)]:
                        (arm/f'{name}.json').write_text(json.dumps(value))
                    for phase in ('train','eval'):
                        path=arm/f'{phase}.json';saved_receipts[phase]=path.read_text()
                        value=json.loads(saved_receipts[phase]);value['calls']=12
                        for key in ('query_gradient_calls','key_gradient_calls','value_gradient_calls'):value[key]=12
                        path.write_text(json.dumps(value))
                    args.output=str(root/'twelve-layer.json');args.schedule_steps=steps;args.warmup_steps=1;args.num_hidden_layers=12
                    validate(args)
                    args.output=str(root/'wrong-depth.json');args.num_hidden_layers=28
                    with self.assertRaises(AssertionError):validate(args)
                    for name,value in [('train_config',config),('trainer_state',state),('result',result)]:
                        (arm/f'{name}.json').write_text(json.dumps(value))
                    for phase,value in saved_receipts.items():(arm/f'{phase}.json').write_text(value)
                    args.schedule_steps=28600;args.warmup_steps=1430
                args.output=str(root/'bad.json');args.steps=steps+1
                with self.assertRaises(AssertionError):validate(args)
                args.steps=steps
                if variant in P6_EXTENSION_ARMS:
                    # A label alone cannot masquerade as the corresponding control.
                    path=arm/'train_config.json';bad=json.loads(path.read_text())
                    bad['proxy_target']['extension']['source']='incorrect';path.write_text(json.dumps(bad))
                    with self.assertRaises(AssertionError):validate(args)
                    path.write_text(json.dumps(config))
                if steps==25 and interval==10:
                    path=arm/'trainer_state.json';broken={**state,'log_history':state['log_history'][:-1]}
                    path.write_text(json.dumps(broken))
                    with self.assertRaises(AssertionError):validate(args)
                    path.write_text(json.dumps(state))
                if backend=='sdpa':
                    path=arm/'train.json';receipt=json.loads(path.read_text());receipt['has_math']=True
                    path.write_text(json.dumps(receipt))
                    with self.assertRaises(AssertionError):validate(args)
                    receipt['has_math']=False;path.write_text(json.dumps(receipt))
                (checkpoint/'rng_state_7.pth').unlink()
                with self.assertRaises(FileNotFoundError):validate(args)
                self.assertFalse((root/'bad.json').exists())


if __name__=='__main__':unittest.main()
