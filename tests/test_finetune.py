"""CPU checks of downstream adaptation, padding, gradients and real Trainer IO."""
import json
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from datasets import Dataset
from safetensors.torch import save_file, load_file
from transformers import TrainingArguments

from eval.finetune import PairClassifier, PairCollator, encode_pairs, metrics, SupervisedTrainer, GradientCheck
from tests.test_eval import make_model
from tests.test_train import fixture
from tests.test_fa4_baseline import reference_kernel
from deep_kv import P6_VARIANTS, P7_P6_VARIANTS


class FinetuneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_forward_identical_and_all_branches_learn(self):
        for backend in ('sdpa', 'fa4'):
            with patch('deep_kv.fa4.load_kernel', return_value=(reference_kernel, {'version': 'CPU-test-double'})):
                for arm in ('A', 'P6', 'P6-iso', 'P7-simple') + P6_VARIANTS + P7_P6_VARIANTS:
                    with self.subTest(backend=backend, arm=arm):
                        backbone = make_model(arm, backend)
                        rows = [dict(input_ids=[2, 4, 3, 31], labels=0),
                                dict(input_ids=[5, 7, 9, 4, 31], labels=1)]
                        inputs = PairCollator(31)(rows)
                        from deep_kv.model import Context
                        positions = torch.arange(8).expand(2, -1)
                        valid = inputs['attention_mask'].bool()
                        pos = torch.where(valid, positions, positions-valid.sum(-1)[:,None])
                        ctx = Context(inputs['input_ids'], torch.ones_like(valid), pos, (~valid).long())
                        old = backbone.hidden_states(ctx)[0].detach()
                        model = PairClassifier(backbone, 2, 42).train()
                        torch.testing.assert_close(backbone.hidden_states(ctx)[0], old, rtol=0, atol=0)
                        output = model(**inputs)
                        torch.testing.assert_close(output['logits'], model.score(old[torch.arange(2),valid.sum(-1)-1]))
                        output['loss'].backward()
                        check = GradientCheck(); check.on_pre_optimizer_step(None,None,None,model=model)
                        self.assertEqual(check.report['status'], 'passed')
                        if arm != 'A': self.assertGreater(check.report['squared_gradient_norms']['proxy'], 0)
                        alone = PairCollator(31)(rows[:1])
                        torch.testing.assert_close(model(**alone)['logits'], output['logits'][:1], rtol=1e-5,atol=1e-6)
                        changed = {k: v.clone() for k,v in inputs.items()}
                        changed['input_ids'][~valid] = 20
                        torch.testing.assert_close(model(**changed)['logits'], output['logits'], rtol=0,atol=0)

    def test_balanced_truncation_and_eos(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            from transformers import AutoTokenizer
            tok=AutoTokenizer.from_pretrained(root/'model',local_files_only=True)
            batch=dict(a=[' '.join(['2']*70),'2'],b=[' '.join(['3']*70),'3'],label=[0,1])
            out=encode_pairs(batch,tok,('a','b'),32)
            self.assertEqual(len(out['input_ids'][0]),32)
            self.assertTrue(2 in out['input_ids'][0] and 3 in out['input_ids'][0])
            self.assertEqual([x[-1] for x in out['input_ids']],[31,31])
            self.assertEqual(out['truncated'],[True,False])

    def test_p6_iso_task_mode_preserves_default_isolation(self):
        model = make_model('P6-iso')
        head = next(iter(model.heads.values()))
        u = torch.randn(2, 4, model.backbone.config.hidden_size, requires_grad=True)
        before, aux = head.routed(u, model.settings)
        self.assertFalse(before.requires_grad)
        aux.sum().backward()
        self.assertIsNone(u.grad)
        self.assertIsNotNone(head.w1.weight.grad)
        head.zero_grad(set_to_none=True)
        PairClassifier(model, 2, 42)
        after, _ = head.routed(u, model.settings)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        after.sum().backward()
        self.assertGreater(u.grad.abs().sum().item(), 0)
        self.assertGreater(head.w1.weight.grad.abs().sum().item(), 0)
        self.assertTrue(model.settings.isolate_estimator)
        self.assertNotIn('task_finetuning', model.state_dict())
        # Reconstructing a pretraining model never inherits downstream mode.
        fresh = make_model('P6-iso')
        fresh.load_state_dict(model.state_dict(), strict=True)
        fresh_head = next(iter(fresh.heads.values()))
        self.assertFalse(fresh_head.routed(u, fresh.settings)[0].requires_grad)

    def test_real_trainer_save_reload_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            model=PairClassifier(make_model('P7-simple'),3,42)
            rows=[dict(input_ids=[2,3,4,31],labels=i%3) for i in range(6)]
            ds=Dataset.from_list(rows)
            check=GradientCheck()
            args=TrainingArguments(output_dir=tmp,use_cpu=True,report_to=[],max_steps=2,
                per_device_train_batch_size=2,per_device_eval_batch_size=2,learning_rate=1e-3,
                eval_strategy='steps',eval_steps=1,save_strategy='steps',save_steps=1,
                load_best_model_at_end=True,metric_for_best_model='accuracy',greater_is_better=True,
                save_total_limit=1,save_only_model=True,disable_tqdm=True)
            trainer=SupervisedTrainer(model=model,args=args,train_dataset=ds,eval_dataset=ds,
                data_collator=PairCollator(31),compute_metrics=metrics,callbacks=[check])
            trainer.train(); trainer.save_model(str(Path(tmp)/'best'))
            self.assertEqual(check.report['status'],'passed')
            prediction=trainer.predict(ds)
            clone=PairClassifier(make_model('P7-simple'),3,42)
            clone.load_state_dict(load_file(Path(tmp)/'best/model.safetensors'),strict=True)
            clone.eval()
            with torch.no_grad(): result=clone(**PairCollator(31)(rows))
            np.testing.assert_allclose(result['logits'].numpy(),prediction.predictions,rtol=1e-5,atol=1e-6)


class EntryTests(unittest.TestCase):
    def test_actual_entry_train_then_test(self):
        import hashlib
        from datasets import ClassLabel
        from eval.finetune import run
        from tests.test_proxy_memory import config
        for arm in ('P7-simple', 'P6-iso'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); fixture(root)
                backbone=make_model(arm)
                source_step=10000 if arm=='P6-iso' else 10800
                checkpoint=root/f'checkpoint-{source_step}';checkpoint.mkdir()
                from transformers import AutoTokenizer
                tok=AutoTokenizer.from_pretrained(root/'model',local_files_only=True);tok.save_pretrained(checkpoint)
                save_file({k:v.clone() for k,v in backbone.state_dict().items()},checkpoint/'model.safetensors')
                (checkpoint/'trainer_state.json').write_text(json.dumps(dict(global_step=source_step)))
                recipe=dict(model_config=config().to_dict(),model=dict(tokenizer_name=str(checkpoint)),
                    data=dict(block_size=32),training=dict(seed=42,bf16=False),
                    pilot=dict(arm=arm,proxy_screen=True,consumer=2,deep_target=8,lm_chunk=3,
                        causal_attention=False,**{'proxy_'+k:v for k,v in asdict(backbone.settings).items()}))
                (root/'train_config.json').write_text(json.dumps(recipe))
                files=[];data_files={};folder=root/'raw/paws';folder.mkdir(parents=True)
                for split,n in [('train',12),('validation',4),('test',5)]:
                    ds=Dataset.from_list([dict(sentence1=f'2 {split} {i}',sentence2='3 4',label=i%2) for i in range(n)])
                    ds=ds.cast_column('label',ClassLabel(names=['0','1']))
                    path=folder/(split+'.parquet');ds.to_parquet(path)
                    rel='paws/'+path.name
                    files.append(dict(path=rel,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                    data_files[split]=[rel]
                manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(schema_version=1,repositories=[
                    dict(path='paws',files=files,tasks=['paws_en'],load_configs=dict(paws_en=dict(loader='parquet',data_files=data_files)))])))
                cfg=dict(checkpoint=str(checkpoint),expected_step=source_step,task='paws',dataset_root=str(root/'raw'),dataset_manifest=str(manifest),
                    max_length=32,preprocessing_num_workers=1,attention_backend='sdpa',output_dir=str(root/'fitted'),
                    use_cpu=True,report_to='none',seed=42,data_seed=42,max_steps=2,learning_rate=1e-3,
                    per_device_train_batch_size=2,per_device_eval_batch_size=2,eval_strategy='steps',eval_steps=1,
                    save_strategy='steps',save_steps=1,load_best_model_at_end=True,metric_for_best_model='accuracy',
                    greater_is_better=True,save_only_model=True,save_total_limit=1,disable_tqdm=True)
                path=root/'args.json'
                path.write_text(json.dumps({**cfg, 'expected_step': 2500}))
                with self.assertRaisesRegex(ValueError, 'source checkpoint'): run(path)
                path.write_text(json.dumps(cfg));run(path)
                result=json.loads((root/'fitted/result.json').read_text())
                self.assertEqual(result['global_step'],2)
                cfg.update(evaluate_run=str(root/'fitted'),output_dir=str(root/'test'),eval_strategy='no',save_strategy='no',
                           load_best_model_at_end=False)
                path.write_text(json.dumps(cfg));run(path)
                scored=json.loads((root/'test/result.json').read_text())
                self.assertEqual(scored['data']['rows'],{'test':5})
                self.assertEqual(scored['weight_sha256'],result['weight_sha256'])
                self.assertEqual(np.load(root/'test/predictions.npz')['logits'].shape,(5,2))

class StudyTests(unittest.TestCase):
    def test_single_arm_queue_matches_original_recipe(self):
        from scripts.finetune_study import make_jobs
        from run_experiments import load_jobs
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root/'old'; new = root/'new'
            make_jobs(old, Path.cwd(), '/data', '/manifest', {'P6':'/P6'})
            jobs = make_jobs(new, Path.cwd(), '/data', '/manifest', {'P6-iso':'/P6-iso'})
            load_jobs(new/'jobs.json')
            self.assertEqual(len(jobs), 21)
            fits = [i for i,j in enumerate(jobs) if j['name'].startswith(('search-', 'confirm-'))]
            tests = [i for i,j in enumerate(jobs) if j['name'].startswith('test-')]
            self.assertEqual(len(fits), 8); self.assertEqual(len(tests), 6)
            self.assertLess(max(fits), min(tests))
            self.assertEqual(jobs[-1]['argv'][-2:], ['--arms', 'P6-iso'])
            for p in (new/'configs').glob('*.json'):
                matched = old/'configs'/p.name.replace('P6-iso', 'P6')
                new_config = json.loads(p.read_text().replace(str(new), str(old)).replace('P6-iso', 'P6'))
                self.assertEqual(new_config, json.loads(matched.read_text()))
            self.assertTrue(all(jobs[i]['gpus'] == list(range(8)) for i in fits+tests))

    def test_dev_only_selection_and_unmatched_rejection(self):
        from scripts.finetune_study import select
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths=[]
            for lr,dev,test in [(1e-5,.6,.9),(3e-5,.7,.5)]:
                p=root/str(lr);p.mkdir();paths.append(str(p))
                r=dict(status='completed',smoke=False,evaluate_run=None,learning_rate=lr,source={'arm':'P6'},
                    data={'task':'nli'},seed=42,global_step=20,world_size=8,max_length=512,attention_backend='fa4',
                    metrics={'eval_accuracy':dev,'test_accuracy':test})
                (p/'result.json').write_text(json.dumps(r))
            select(paths,root/'choice.json')
            self.assertEqual(json.loads((root/'choice.json').read_text())['learning_rate'],3e-5)
            r['seed']=43;(Path(paths[1])/'result.json').write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'Unmatched'):
                select(paths,root/'invalid.json')

    def test_queue_counts_and_test_after_all_fitting(self):
        from scripts.finetune_study import make_jobs
        from run_experiments import load_jobs
        with tempfile.TemporaryDirectory() as tmp:
            jobs=make_jobs(tmp,Path.cwd(),'/data','/manifest',dict(A='/A',P6='/P6',**{'P7-simple':'/P7'}))
            load_jobs(Path(tmp)/'jobs.json')
            fits=[i for i,j in enumerate(jobs) if j['name'].startswith(('search-','confirm-'))]
            tests=[i for i,j in enumerate(jobs) if j['name'].startswith('test-')]
            self.assertEqual(len(fits),24);self.assertEqual(len(tests),18)
            self.assertLess(max(fits),min(tests))
            self.assertTrue(all(jobs[i]['gpus']==list(range(8)) for i in fits))


if __name__ == '__main__': unittest.main()
