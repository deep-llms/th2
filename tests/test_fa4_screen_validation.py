"""CPU checks for the production-step checkpoint validator, without GPU work."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from safetensors.torch import save_file
from scripts.check_fa4_proxy import validate


class ScreenValidationTests(unittest.TestCase):
    def test_smoke_and_screen_cutoffs_and_missing_rank_state(self):
        for steps,interval in ((3,1),(2500,10)):
            with self.subTest(steps=steps),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);arm=root/'seed-42/P3-block';checkpoint=arm/f'checkpoint-{steps}'
                checkpoint.mkdir(parents=True)
                config=dict(pilot=dict(attention_backend='fa4',checkpoint_layers=False,checkpoint_lm=False,checkpoint_aux=False),
                    data=dict(isolate_documents=True,eval_rows=32),world_size=8,
                    training=dict(per_device_train_batch_size=16,gradient_accumulation_steps=4,warmup_steps=1430,seed=42,data_seed=42))
                state=dict(global_step=steps,max_steps=28600,log_history=[dict(step=i,loss=3.,grad_norm=.2) for i in range(interval,steps+1,interval)])
                result=dict(arm='P3-block',global_step=steps,status='stopped',schedule_steps=28600,input_tokens=steps*1048576,
                    attention_runtime=dict(version='4.0.0b33'),sdpa_receipts=[],fa4_receipts=['train.json','eval.json'],
                    evaluation=dict(eval_lm_loss=3.,eval_rows=32),training_cost=dict(peak_cuda_allocated_bytes=128*2**30))
                for name,value in (('train_config',config),('trainer_state',state),('result',result)):
                    (arm/f'{name}.json').write_text(json.dumps(value))
                (checkpoint/'trainer_state.json').write_text(json.dumps(state))
                for phase in ('train','eval'):
                    receipt=dict(arm='P3-block',phase=phase,calls=28,world_size=8,kernel_dtype='torch.bfloat16',
                        positions='reset_per_document',document_isolation=True,causal=True,dense_mask=False,
                        query_gradient_calls=28,key_gradient_calls=28,value_gradient_calls=28)
                    (arm/f'{phase}.json').write_text(json.dumps(receipt))
                for name in ['optimizer.pt','scheduler.pt',*[f'rng_state_{i}.pth' for i in range(8)]]:
                    (checkpoint/name).write_bytes(b'fixture')
                save_file(dict(mu_initialized=torch.tensor(True),mu=torch.ones(2,3),sigma2=torch.ones(2,3),
                    **{'heads.2.alpha':torch.ones(1,3)*.1}),checkpoint/'model.safetensors')
                args=SimpleNamespace(run_dir=str(root),output=str(root/'passed.json'),steps=steps,seed=42,arms=['P3-block'])
                validate(args);self.assertEqual(json.loads((root/'passed.json').read_text())['steps'],steps)
                args.output=str(root/'bad.json');args.steps=steps+1
                with self.assertRaises(AssertionError):validate(args)
                args.steps=steps;(checkpoint/'rng_state_7.pth').unlink()
                with self.assertRaises(FileNotFoundError):validate(args)
                self.assertFalse((root/'bad.json').exists())


if __name__=='__main__':unittest.main()
