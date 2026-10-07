"""Regression and BoolQ protocols through real data, Trainer and saved checkpoints."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from datasets import ClassLabel, Dataset, DatasetDict
from safetensors.torch import save_file
from transformers import AutoTokenizer

from eval.finetune import PairClassifier, PairCollator, metrics, supervised_splits, document_key, run
from tests.test_eval import make_model
from tests.test_train import fixture
from tests.test_proxy_memory import config


class ExtendedFinetuneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_regression_mse_float_labels_and_correlations(self):
        for arm in ('A','P6','P6-iso','P7-simple'):
            model=PairClassifier(make_model(arm),1,42)
            inputs=PairCollator(31,regression=True)([dict(input_ids=[2,3,31],labels=.5),dict(input_ids=[4,5,31],labels=3.5)])
            self.assertEqual(inputs['labels'].dtype,torch.float32)
            result=model(**inputs)
            torch.testing.assert_close(result['loss'],((result['logits'][:,0]-inputs['labels'])**2).mean())
            result['loss'].backward()
            from eval.finetune import GradientCheck
            check=GradientCheck();check.on_pre_optimizer_step(None,None,None,model=model)
            self.assertEqual(check.report['status'],'passed')
        actual=metrics((np.array([[0.],[2.],[1.],[3.]]),np.arange(4.)))
        self.assertAlmostEqual(actual['pearson'],.8);self.assertAlmostEqual(actual['spearman'],.8)
        self.assertEqual(metrics((np.ones((4,1)),np.arange(4.)))['correlation'],0.)

    def test_fp32_reference_runs_all_task_gradients_and_restores_state(self):
        from scripts.finetune_study import numerical_check, check_numerical_limits
        from tests.test_fa4_baseline import reference_kernel
        with patch('deep_kv.fa4.load_kernel',return_value=(reference_kernel,{'version':'CPU-test-double'})):
            for arm in ('A','P6','P6-iso','P7-simple'):
                for regression in (False,True):
                    with self.subTest(arm=arm,regression=regression):
                        model=PairClassifier(make_model(arm,'fa4'),1 if regression else 2,42)
                        inputs=PairCollator(31,regression=regression)([
                            dict(input_ids=[2,3,31],labels=.5 if regression else 0),
                            dict(input_ids=[4,5,6,31],labels=3.5 if regression else 1)])
                        before={k:v.clone() for k,v in model.state_dict().items()}
                        flags=torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
                        report=numerical_check(model,inputs)
                        check_numerical_limits(report)
                        self.assertLess(report['logit_relative_l2'],1e-5)
                        self.assertLess(report['gradient_relative_l2'],1e-5)
                        self.assertEqual(report['reference_gradients']['status'],'passed')
                        self.assertEqual(model.wrapped.attention_backend,'fa4')
                        self.assertEqual(flags,(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32))
                        for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
                        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_numerical_limits_reject_errors_and_nonfinite_results(self):
        from scripts.finetune_study import check_numerical_limits
        # Observed A/BoolQ FA4 versus FP32; BF16 SDPA was the inaccurate reference.
        good=dict(logit_relative_l2=.01067456,gradient_relative_l2=.01144959,
                  hidden_relative_l2=.00469895,logit_max_absolute=.00426087,
                  loss_absolute=.00013352,fa4_calls=28,buffers_unchanged=True,task_kind='regression')
        check_numerical_limits(good)
        for key,value in [('logit_relative_l2',.02),('gradient_relative_l2',.05),
                          ('hidden_relative_l2',.02),('fa4_calls',0),('buffers_unchanged',False),
                          ('logit_relative_l2',float('nan')),('gradient_relative_l2',float('inf')),
                          ('loss_absolute',float('nan'))]:
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                check_numerical_limits(dict(good,**{key:value}))

    def test_classification_gate_is_offset_invariant_and_rejects_probability_errors(self):
        from scripts.finetune_study import classification_difference, check_numerical_limits
        ref=torch.tensor([[.125232905,.179020077],[.156042695,.437513232]],dtype=torch.float64)
        actual=torch.tensor([[.13671875,.181640625],[.169921875,.443359375]],dtype=torch.float64)
        base=dict(task_kind='classification',gradient_relative_l2=.0150184,hidden_relative_l2=.00825454,
                  logit_relative_l2=.03724899,logit_max_absolute=.01387918,loss_absolute=.00389302,
                  fa4_calls=40,buffers_unchanged=True)
        score=classification_difference(actual,ref)
        self.assertLess(score['probability_max_absolute'],.003)
        check_numerical_limits(dict(base,**score))
        shifted=classification_difference(actual+torch.tensor([[100.],[-100.]]),ref)
        self.assertAlmostEqual(score['probability_max_absolute'],shifted['probability_max_absolute'],places=12)
        # A genuine class-specific perturbation must fail, even with good gradients.
        bad=actual.clone();bad[:,0]+=.2
        with self.assertRaises(ValueError):check_numerical_limits(dict(base,**classification_difference(bad,ref)))
        for bad_metric,value in [('probability_max_absolute',.01),('loss_absolute',.01),
                                 ('probability_max_absolute',float('nan'))]:
            with self.subTest(metric=bad_metric),self.assertRaises(ValueError):
                check_numerical_limits({**base,**score,bad_metric:value})

    def test_grouped_split_is_fixed_disjoint_and_ignores_hidden_test(self):
        rows=[dict(passage=f'a {i}',question=f'b {i}',label=i%2) for i in range(40)]
        rows+=[dict(passage='a 3',question='b 3',label=1)]
        raw=DatasetDict(train=Dataset.from_list(rows),validation=Dataset.from_list(rows[:2]),
                        test=Dataset.from_list([dict(unused='hidden labels never used')]))
        split=supervised_splits(raw,'boolq')
        groups=[{document_key(r,('passage','question'),False) for r in split[s]} for s in ('train','validation','test')]
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        changed=DatasetDict(train=raw['train'],validation=raw['validation'].map(lambda row:{'label':1-row['label']}))
        again=supervised_splits(changed,'boolq')
        self.assertEqual(split['train'].to_dict(),again['train'].to_dict())
        self.assertEqual(split['validation'].to_dict(),again['validation'].to_dict())

    def test_stsb_uses_official_dev_and_removes_dev_test_pair_overlap(self):
        raw=DatasetDict(train=Dataset.from_list([dict(sentence1='a',sentence2='b',score=.5)]),
            validation=Dataset.from_list([dict(sentence1='d',sentence2='e',score=.4),dict(sentence1='g',sentence2='f',score=.2)]),
            test=Dataset.from_list([dict(sentence1='f',sentence2='g',score=.2)]))
        out=supervised_splits(raw,'stsb')
        self.assertEqual(out['train'].to_dict(),raw['train'].to_dict())
        self.assertEqual(out['test'].to_dict(),raw['test'].to_dict())
        self.assertEqual(out['validation']['sentence1'],['d'])

    def test_actual_regression_and_boolq_train_reload(self):
        for task,arm in [('stsb','P7-simple'),('boolq','P6-iso')]:
            with self.subTest(task=task),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);fixture(root);backbone=make_model(arm)
                checkpoint=root/'checkpoint-2500';checkpoint.mkdir()
                tok=AutoTokenizer.from_pretrained(root/'model',local_files_only=True);tok.save_pretrained(checkpoint)
                save_file({k:v.clone() for k,v in backbone.state_dict().items()},checkpoint/'model.safetensors')
                (checkpoint/'trainer_state.json').write_text(json.dumps(dict(global_step=2500)))
                recipe=dict(model_config=config().to_dict(),model=dict(tokenizer_name=str(checkpoint)),
                    data=dict(block_size=32),training=dict(seed=42,bf16=False),
                    pilot=dict(arm=arm,proxy_screen=True,consumer=2,deep_target=8,lm_chunk=3,causal_attention=False,
                               **{'proxy_'+k:v for k,v in asdict(backbone.settings).items()}))
                (root/'train_config.json').write_text(json.dumps(recipe))
                fields=('sentence1','sentence2') if task=='stsb' else ('passage','question')
                raw=root/'raw';raw.mkdir();files=[];splits={}
                for split,n in ([('train',40),('validation',8),('test',9)] if task=='stsb' else [('train',40),('validation',9)]):
                    ds=Dataset.from_list([{fields[0]:f'2 {split} {i}',fields[1]:f'3 {i%8}',
                                          ('score' if task=='stsb' else 'label'):float(i%6)/5 if task=='stsb' else i%2} for i in range(n)])
                    if task=='boolq':ds=ds.cast_column('label',ClassLabel(names=['False','True']))
                    file=raw/(split+'.parquet');ds.to_parquet(file)
                    files.append(dict(path=file.name,bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
                    splits[split]=[file.name]
                manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(schema_version=1,repositories=[
                    dict(path='.',tasks=[task],files=files,load_configs={task:dict(loader='parquet',data_files=splits)})])))
                cfg=dict(checkpoint=str(checkpoint),task=task,dataset_root=str(raw),dataset_manifest=str(manifest),
                    max_length=32,preprocessing_num_workers=1,attention_backend='sdpa',output_dir=str(root/'fitted'),
                    use_cpu=True,report_to='none',seed=42,data_seed=42,max_steps=2,learning_rate=1e-3,
                    per_device_train_batch_size=2,per_device_eval_batch_size=2,eval_strategy='steps',eval_steps=1,
                    save_strategy='steps',save_steps=1,load_best_model_at_end=True,
                    metric_for_best_model='correlation' if task=='stsb' else 'accuracy',greater_is_better=True,
                    save_only_model=True,save_total_limit=1,disable_tqdm=True,smoke=True)
                path=root/'args.json';path.write_text(json.dumps(cfg));run(path)
                saved=json.loads((root/'fitted/result.json').read_text())
                cfg.update(evaluate_run=str(root/'fitted'),output_dir=str(root/'tested'),eval_strategy='no',save_strategy='no',load_best_model_at_end=False)
                path.write_text(json.dumps(cfg));run(path)
                final=json.loads((root/'tested/result.json').read_text())
                from scripts.finetune_study import verify_reload
                verify_reload(root/'fitted',root/'tested',root/'reload-check.json')
                cfg.update(smoke=False,output_dir=str(root/'final'))
                path.write_text(json.dumps(cfg));run(path)
                final=json.loads((root/'final/result.json').read_text())
                self.assertEqual(final['weight_sha256'],saved['weight_sha256'])
                self.assertEqual(final['data']['final_source_split'],'test' if task=='stsb' else 'validation')
                self.assertEqual(final['data']['rows'],{'test':9})
                p=np.load(root/'final/predictions.npz')
                self.assertEqual(p['logits'].shape,(9,1 if task=='stsb' else 2))
                for k,v in metrics((p['logits'],p['labels'])).items():self.assertAlmostEqual(final['metrics']['test_'+k],v)

    def test_final_summary_recomputes_regression_metrics(self):
        from scripts.finetune_study import summary
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'selections').mkdir()
            (root/'selections/stsb-A.json').write_text(json.dumps({'learning_rate':1e-5}))
            labels=np.linspace(0,5,1379,dtype=np.float32);logits=(labels*.8+.1)[:,None]
            for seed in (42,43,44):
                folder=root/f'test/stsb/A/seed-{seed}';folder.mkdir(parents=True)
                np.savez(folder/'predictions.npz',logits=logits,labels=labels)
                data=dict(task='stsb',rows={'test':1379},train_order_sha256='train',dataset_manifest_sha256='manifest',
                          tokenizer_sha256='tokenizer',split_order_sha256={'test':'order'})
                result=dict(status='completed',source={'arm':'A'},data=data,seed=seed,smoke=False,world_size=8,
                            attention_backend='fa4',learning_rate=1e-5,metrics={'test_'+k:v for k,v in metrics((logits,labels)).items()})
                (folder/'result.json').write_text(json.dumps(result))
            summary(root,root/'summary.json',arms=['A'],tasks=['stsb'])
            value=json.loads((root/'summary.json').read_text())['tasks']['stsb']['A']
            self.assertAlmostEqual(value['metrics']['pearson']['mean'],1.)
            result['metrics']['test_pearson']=.4
            (folder/'result.json').write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,'Metric mismatch'):
                summary(root,root/'bad.json',arms=['A'],tasks=['stsb'])

    def test_extended_queue_tasks_gates_and_selection(self):
        from scripts.finetune_study import make_jobs,select
        from run_experiments import load_jobs
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);jobs=make_jobs(root/'study',Path.cwd(),'/data','/manifest',dict(A='/A',P6='/P6',**{'P6-iso':'/iso','P7-simple':'/P7'}),tasks=('stsb','boolq'))
            load_jobs(root/'study/jobs.json')
            self.assertEqual(len(jobs),97)
            gates=[j for j in jobs if j['name'].startswith('gate-')]
            self.assertEqual(len(gates),8)
            for job in gates:
                self.assertIn('--data-root',job['argv']);self.assertIn('--manifest',job['argv'])
            fits=[i for i,j in enumerate(jobs) if j['name'].startswith(('search-','confirm-'))]
            tests=[i for i,j in enumerate(jobs) if j['name'].startswith('test-')]
            self.assertEqual(len(fits),32);self.assertEqual(len(tests),24);self.assertLess(max(fits),min(tests))
            paths=[]
            for lr,score in [(1e-5,.8),(3e-5,.7)]:
                p=root/str(lr);p.mkdir();paths.append(str(p))
                d=dict(status='completed',smoke=False,evaluate_run=None,learning_rate=lr,source={'arm':'A'},data={'task':'stsb'},seed=42,
                       global_step=30,world_size=8,max_length=512,attention_backend='fa4',metrics={'eval_correlation':score})
                (p/'result.json').write_text(json.dumps(d))
            select(paths,root/'selection.json');self.assertEqual(json.loads((root/'selection.json').read_text())['learning_rate'],1e-5)


if __name__=='__main__':unittest.main()
