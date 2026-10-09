"""CPU checks for the production-step checkpoint validator, without GPU work."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from safetensors.torch import save_file
from scripts.check_fa4_proxy import validate
from deep_kv import MEMORY_ARMS, SIMPLE_MEMORY_ARMS


class ScreenValidationTests(unittest.TestCase):
    def test_smoke_and_screen_cutoffs_and_missing_rank_state(self):
        for backend,variant,steps,interval in (('fa4','P3-block',3,1),('fa4','P3-block',2500,10),
                                             ('fa4','P6-iso-sparse',25,10),('fa4','P7-simple-short',25,10),
                                             ('fa4','P4-4h',25,1),('fa4','P4-iso-4h',2500,10),
                                             ('sdpa','P4-4h',3,1),('sdpa','P4-iso-4h',2500,10),
                                             ('fa4','P7',25,1),('fa4','P7-kq',25,1),('sdpa','P7-ems',25,1),
                                             ('fa4','P7-simple',25,1),('sdpa','P7-simple',25,1),
                                             ('fa4','P7-simple-sparse',25,1),('sdpa','P7-simple-sparse',25,1),
                                             ('fa4','P7-simple-short',25,1),('sdpa','P7-simple-short',25,1)):
            with self.subTest(steps=steps,arm=variant),tempfile.TemporaryDirectory() as tmp:
                layers=(2,6,10,14,18,22) if variant=='P7-simple-sparse' else tuple(range(2,25,2))
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
                for name,value in (('train_config',config),('trainer_state',state),('result',result)):
                    (arm/f'{name}.json').write_text(json.dumps(value))
                (checkpoint/'trainer_state.json').write_text(json.dumps(state))
                for phase in ('train','eval'):
                    calls=28+len(layers) if variant in MEMORY_ARMS else 28
                    receipt=dict(arm=variant,phase=phase,calls=calls,world_size=8,kernel_dtype='torch.bfloat16',
                        positions='reset_per_document',document_isolation=True,causal=True,dense_mask=False,
                        query_gradient_calls=calls,key_gradient_calls=calls,value_gradient_calls=calls)
                    if backend=='sdpa':
                        receipt.update(calls=[dict(mask=dict(dtype='torch.bool'),is_causal=False,backward_operators=['fused_backward'])]*calls,
                            has_math=False,unattributed_calls=[],unattributed_backward=[],
                            checkpointing=dict(checkpoint_layers=False,checkpoint_lm=False,checkpoint_aux=False))
                    (arm/f'{phase}.json').write_text(json.dumps(receipt))
                for name in ['optimizer.pt','scheduler.pt',*[f'rng_state_{i}.pth' for i in range(8)]]:
                    (checkpoint/name).write_bytes(b'fixture')
                heads=({f'heads.{layer}.{suffix}':torch.ones(3) for layer in layers
                        for suffix in (('w1.weight','w2.weight') if variant in SIMPLE_MEMORY_ARMS else ('conv','w_in.weight','w_out.weight'))+('k_proj.weight','k_norm.weight')+
                        (() if variant=='P7-kq' else ('v_proj.weight',))} if variant in MEMORY_ARMS else
                       {'heads.2.alpha':torch.ones(1,3)*.1})
                save_file(dict(mu_initialized=torch.tensor(True),mu=torch.ones(2,3),sigma2=torch.ones(2,3),
                    **heads),checkpoint/'model.safetensors')
                args=SimpleNamespace(run_dir=str(root),output=str(root/'passed.json'),steps=steps,seed=42,arms=[variant],attention_backend=backend)
                validate(args);self.assertEqual(json.loads((root/'passed.json').read_text())['steps'],steps)
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
                args.output=str(root/'bad.json');args.steps=steps+1
                with self.assertRaises(AssertionError):validate(args)
                args.steps=steps
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
