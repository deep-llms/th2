"""Training-recipe compatibility and complete native Trainer checkpoint guards."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from test_document_isolation import model, torch
from train import BaselineTrainingArguments
from resume_recipe import (build_training_recipe, differences, require_resume_training,
                           save_training_recipe, fp32_precision_settings, FILENAME)


class ResumeRecipeTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(torch.backends, 'fp32_precision'), 'Requires new precision API')
    def test_independent_cudnn_precision_is_read_only_and_resume_checked(self):
        conv, rnn = torch.backends.cudnn.conv, torch.backends.cudnn.rnn
        original = conv.fp32_precision, rnn.fp32_precision
        try:
            conv.fp32_precision, rnn.fp32_precision = 'ieee', 'tf32'
            with self.assertRaises(RuntimeError):
                _ = torch.backends.cudnn.allow_tf32
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);m=model();args=self.args(root)
                before=fp32_precision_settings(torch)
                recipe=build_training_recipe(m,args,'sdpa')
                self.assertEqual(before,fp32_precision_settings(torch))
                self.assertEqual(recipe['execution']['fp32_precision']['cudnn_conv'],'ieee')
                self.assertEqual(recipe['execution']['fp32_precision']['cudnn_rnn'],'tf32')
                self.checkpoint(root,recipe)
                rnn.fp32_precision='ieee'
                with self.assertRaisesRegex(ValueError,'cudnn_rnn'):
                    require_resume_training(root,build_training_recipe(m,args,'sdpa'),args)
        finally:
            conv.fp32_precision, rnn.fp32_precision = original

    def test_old_runtime_precision_reader(self):
        fake=SimpleNamespace(get_float32_matmul_precision=lambda:'highest',
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                                     cudnn=SimpleNamespace(allow_tf32=True)))
        self.assertEqual(fp32_precision_settings(fake),dict(float32_matmul_precision='highest',
            cuda_matmul_allow_tf32=False,cudnn_allow_tf32=True))

    def args(self, root, **changes):
        return BaselineTrainingArguments(**dict(output_dir=str(root),use_cpu=True,report_to='none',
            max_steps=8,stop_after=6,per_device_train_batch_size=1,gradient_accumulation_steps=2,
            learning_rate=.001,lr_scheduler_type='cosine',warmup_ratio=.5,
            seed=42,data_seed=123,dataloader_pin_memory=False) | changes)

    def checkpoint(self, root, recipe):
        save_training_recipe(root,recipe)
        (root/'trainer_state.json').write_text(json.dumps({'global_step':3,'max_steps':8,
            'train_batch_size':recipe['execution']['train_batch_size_per_process']}))
        world=recipe['execution']['world_size']
        files=['optimizer.pt','scheduler.pt']+(['rng_state.pth'] if world==1 else
                                            [f'rng_state_{rank}.pth' for rank in range(world)])
        for name in files:(root/name).write_bytes(b'presence fixture; actual loading is tested with Trainer')

    def test_operational_changes_allowed_and_scientific_changes_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m=model();args=self.args(root)
            original=build_training_recipe(m,args,'sdpa');self.checkpoint(root,original)
            changed=self.args(root/'new-output',stop_after=8,save_steps=2,save_total_limit=None,
                              logging_steps=1,logging_strategy='no',disable_tqdm=True)
            actual=build_training_recipe(m,changed,'sdpa')
            self.assertEqual(original,actual)
            require_resume_training(root,actual,changed)
            for field,value in [('learning_rate',.002),('warmup_ratio',.25),('lr_scheduler_type','linear'),
                    ('weight_decay',.1),('adam_beta1',.8),('adam_epsilon',1e-6),('max_grad_norm',.5),
                    ('optim','sgd'),('seed',9),('data_seed',9),('per_device_train_batch_size',2),
                    ('gradient_accumulation_steps',1),('gradient_checkpointing',True),('bf16',True),
                    ('dataloader_drop_last',True),('dataloader_num_workers',1),('full_determinism',True)]:
                with self.subTest(field=field):
                    candidate=build_training_recipe(m,self.args(root,**{field:value}),'sdpa')
                    with self.assertRaisesRegex(ValueError,field):require_resume_training(root,candidate,args)
            for section,field,value in [('execution','world_size',2),('runtime','torch','changed'),
                                        ('model','attention_backend','fa4')]:
                candidate=copy.deepcopy(original);candidate[section][field]=value
                with self.assertRaisesRegex(ValueError,field):require_resume_training(root,candidate,args)
            candidate=build_training_recipe(m,args,'sdpa',SimpleNamespace(mixed_precision='bf16'))
            with self.assertRaisesRegex(ValueError,'mixed_precision'):
                require_resume_training(root,candidate,args)
            m.config.attention_dropout=.2
            with self.assertRaisesRegex(ValueError,'attention_dropout'):
                require_resume_training(root,build_training_recipe(m,args,'sdpa'),args)

    def test_missing_receipt_state_and_rng_are_not_silent_fresh_optimizers(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);args=self.args(root);recipe=build_training_recipe(model(),args,'sdpa')
            with self.assertRaisesRegex(ValueError,'lacks training_recipe'):require_resume_training(root,recipe,args)
            self.checkpoint(root,recipe)
            require_resume_training(root,recipe,args)
            for name in ('optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json'):
                p=root/name;raw=p.read_bytes();p.unlink()
                with self.assertRaisesRegex(ValueError,name):require_resume_training(root,recipe,args)
                p.write_bytes(raw)
            (root/'optimizer.pt').write_bytes(b'')
            with self.assertRaisesRegex(ValueError,'optimizer.pt'):require_resume_training(root,recipe,args)
            self.checkpoint(root,recipe)
            for key,value in [('schema',99),('schema',None)]:
                bad=copy.deepcopy(recipe);bad[key]=value;save_training_recipe(root,bad)
                with self.assertRaisesRegex(ValueError,'schema'):require_resume_training(root,recipe,args)
            self.checkpoint(root,recipe)
            (root/'trainer_state.json').write_text(json.dumps({'global_step':8,'max_steps':8,'train_batch_size':1}))
            with self.assertRaisesRegex(ValueError,'already reached'):
                require_resume_training(root,recipe,self.args(root,stop_after=None))
            state={'train_batch_size':2};(root/'trainer_state.json').write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError,'effective train batch size'):require_resume_training(root,recipe,args)

    def test_every_distributed_rank_rng_is_required(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);args=self.args(root);recipe=build_training_recipe(model(),args,'sdpa')
            recipe['execution']['world_size']=2
            recipe['execution']['global_batch_size']*=2
            self.checkpoint(root,recipe)
            require_resume_training(root,recipe,args)
            (root/'rng_state_1.pth').unlink()
            with self.assertRaisesRegex(ValueError,'rng_state_1'):require_resume_training(root,recipe,args)

    def test_effective_fp16_requires_scaler_even_without_argument_override(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);args=self.args(root);recipe=build_training_recipe(model(),args,'sdpa')
            # File-contract simulation only, not a GPU/FP16 numerical test.
            recipe['execution'].update(device_type='cuda',mixed_precision='fp16')
            self.checkpoint(root,recipe)
            with self.assertRaisesRegex(ValueError,'scaler.pt'):require_resume_training(root,recipe,args)
            (root/'scaler.pt').write_bytes(b'presence fixture')
            require_resume_training(root,recipe,args)

    def test_default_data_seed_and_model_metadata_are_semantic(self):
        with tempfile.TemporaryDirectory() as d:
            m=model();a=build_training_recipe(m,self.args(d,data_seed=None),'sdpa')
            m.config._name_or_path='/relocated/initial/assets'
            b=build_training_recipe(m,self.args(d,data_seed=42),'sdpa')
            self.assertEqual(a,b)
            next(m.parameters()).requires_grad_(False)
            c=build_training_recipe(m,self.args(d,data_seed=42),'sdpa')
            self.assertIn('model.tensor_schema_sha256',differences(b,c))
            # Unknown Trainer options remain compared, never automatically exempt.
            b['training']['future_optimizer_option']=1
            self.assertIn('training.future_optimizer_option',differences(a,b))


if __name__=='__main__':unittest.main()
