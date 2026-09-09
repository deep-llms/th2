import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments, default_data_collator, set_seed

from capacity_allocation.modeling import ARMS, SHARED_ARMS, build_model, experiment_config
from capacity_allocation.data import preprocess_text, load_text_data
from eval.diagnostic_data import BUCKETS, count_tokens, frequency_buckets, load_bundle, prepare
from eval.diagnostic_metrics import (centered_covariance, embedding_spectra, frequency_nll,
                                     interfaces, spectrum_report, target_losses)
from eval.diagnostic_gradients import probe_gradients, row_statistics, path_comparison
from eval.diagnostic_training import attach_training_diagnostics
from eval.diagnostics_checkpoint import main as diagnostic_main, training_provenance
from eval.compare_diagnostics import compare
from test_capacity_data_train import fixture
from train import CausalTrainer


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_bucket_ties_special_unseen_and_small_vocab(self):
        counts = np.ones(105, dtype=np.int64)
        counts[100:] = 0
        buckets = frequency_buckets(counts, eos_id=103, pad_id=104)
        self.assertEqual(np.bincount(buckets).tolist(), [1,9,40,40,10,3,1,1])
        self.assertEqual(buckets[0], 0)
        self.assertEqual(buckets[99], 4)
        self.assertEqual(frequency_buckets(np.array([10,0]), 0, 0).tolist(), [6,5])
        with self.assertRaises(ValueError):
            frequency_buckets(np.array([-1, 0]))

    def test_shift_mask_and_eos(self):
        torch.manual_seed(42)
        logits = torch.randn(2,5,9, requires_grad=True)
        batch = dict(input_ids=torch.tensor([[0,1,2,3,4],[1,2,3,4,5]]),
                     labels=torch.tensor([[0,1,-100,3,4],[1,2,3,4,5]]),
                     attention_mask=torch.tensor([[1,1,1,1,1],[1,1,1,0,0]]))
        losses, targets = target_losses(logits, batch, chunk_tokens=2)
        self.assertEqual(targets.tolist(), [1,3,4,2,3])
        expected = torch.nn.functional.cross_entropy(logits[...,:-1,:].transpose(1,2),
            batch['labels'][:,1:], reduction='none')
        mask = batch['labels'][:,1:].ne(-100) & batch['attention_mask'][:,1:].bool()
        torch.testing.assert_close(losses, expected[mask])
        losses.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_covariance_direct_svd_constant_and_empty(self):
        torch.manual_seed(4)
        table = torch.randn(27,7, dtype=torch.float64)+1000
        centered = table-table.mean(0)
        cov = centered_covariance(table, np.arange(27), chunk_rows=3)
        torch.testing.assert_close(cov, centered.T@centered/27, atol=1e-11, rtol=1e-11)
        result = spectrum_report(cov,27,7)
        torch.testing.assert_close(torch.tensor(result['singular_values'],dtype=torch.float64),
                                  torch.linalg.svdvals(centered), atol=1e-10, rtol=1e-10)
        self.assertEqual(result['numerical_rank'], 7)
        const = centered_covariance(torch.ones(8,5), np.arange(8),2)
        self.assertEqual(spectrum_report(const,8,5)['effective_rank'],0)
        self.assertEqual(spectrum_report(None,0,5)['status'],'empty_subset')

    def test_all_arm_interface_orientation_and_functional_spectra(self):
        torch.manual_seed(42)
        ids = torch.arange(97)
        counts = np.ones(97,dtype=np.int64)
        counts[80:]=0
        mapping = frequency_buckets(counts,96,96)
        for arm in ARMS:
            model=build_model(experiment_config(arm,tiny=True)).eval()
            report=embedding_spectra(model,counts,mapping,chunk_rows=13)
            for side,(table,adapter) in interfaces(model).items():
                dense = table.detach().double()
                if adapter is not None:
                    dense = dense@adapter.detach().double()
                if side=='input':
                    torch.testing.assert_close(model.get_input_embeddings()(ids).double(), dense, rtol=1e-5,atol=1e-7)
                else:
                    hidden=torch.randn(2,dense.shape[1])
                    torch.testing.assert_close(model.get_output_embeddings()(hidden).double(),
                                              hidden.double()@dense.T, rtol=1e-5,atol=1e-7)
                eig = spectrum_report(centered_covariance(dense,np.arange(80),11),80,min(dense.shape))
                actual=report['interfaces'][side]['effective']['seen']
                np.testing.assert_allclose(actual['covariance_eigenvalues'],eig['covariance_eigenvalues'],atol=1e-12,rtol=1e-6)
                self.assertLessEqual(actual['numerical_rank'],actual['centered_rank_ceiling'])

    def test_row_rms_empty_and_active_cosines(self):
        g=torch.tensor([[2.,2.,2.,2.],[0.,0.,0.,0.]])
        result=row_statistics(g,np.array([0,0]))
        self.assertAlmostEqual(result['head']['mean_row_grad_rms'],1.)
        self.assertEqual(result['head']['zero_fraction'],.5)
        self.assertIsNone(result['tail']['median_row_grad_rms'])
        out=-g.clone()
        report=path_comparison(g+out,g,out,np.array([0,0]),1e-5)
        self.assertEqual(report['overall']['cosine'],-1.)
        self.assertEqual(report['overall']['active_input_rows'],1)
        self.assertEqual(report['overall']['active_rows_cosine'],-1.)
        with self.assertRaises(ValueError):
            path_comparison(torch.ones_like(g),g,out,np.array([0,0]),1e-5)

    def test_frozen_bundle_counts_hash_and_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); tokenizer=fixture(root)
            manifest=prepare(root/'train',root/'eval',root/'tokenizer',root/'bundle',vocab_size=97,
                block_size=8,workers=1,map_batch_size=4,probe_blocks=2)
            loaded,counts,mapping,data=load_bundle(root/'bundle',tokenizer,97)
            train=preprocess_text(load_text_data(root/'train'),tokenizer,block_size=8,num_proc=1,batch_size=4)
            np.testing.assert_array_equal(counts,count_tokens(train,97))
            self.assertEqual(counts.sum(),len(train)*8)
            self.assertEqual(manifest['probe_ids'],loaded['probe_ids'])
            checkpoint=root/'run/checkpoint-1'; checkpoint.mkdir(parents=True)
            (checkpoint/'trainer_state.json').write_text(json.dumps({'global_step':1}))
            config=dict(train_fingerprint=loaded['training']['shuffled_fingerprint'],
                eval_fingerprint=loaded['validation']['en']['packed_fingerprint'],
                execution={'effective_batch_size':2},data=dict(languages='en',block_size=8,
                    preprocessing_num_workers=1,preprocessing_batch_size=4))
            (checkpoint.parent/'train_config.json').write_text(json.dumps(config))
            self.assertEqual(training_provenance(checkpoint,loaded)['consumed_input_tokens'],16)
            config['train_fingerprint']='different'
            (checkpoint.parent/'train_config.json').write_text(json.dumps(config))
            with self.assertRaises(ValueError): training_provenance(checkpoint,loaded)
            with self.assertRaises(FileExistsError):
                prepare(root/'train',root/'eval',root/'tokenizer',root/'bundle',vocab_size=97)
            with (root/'bundle/frequencies.npz').open('ab') as handle: handle.write(b'corrupt')
            with self.assertRaises(ValueError): load_bundle(root/'bundle')

    def test_bundle_matches_real_training_cli_checkpoint(self):
        from train import main as train_main
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); fixture(root)
            argv=['train','--tokenizer_name',str(root/'tokenizer'),'--arm','B0','--tiny_test','true',
                '--data_dir',str(root/'train'),'--eval_data_dir',str(root/'eval'),
                '--block_size','8','--preprocessing_num_workers','1','--preprocessing_batch_size','4',
                '--output_dir',str(root/'run'),'--use_cpu','true','--per_device_train_batch_size','2',
                '--num_train_epochs','1','--stop-at-step','1','--save_steps','1',
                '--report_to','none','--disable_tqdm','true']
            with patch.object(sys,'argv',argv): train_main()
            manifest=prepare(root/'train',root/'eval',root/'tokenizer',root/'bundle',vocab_size=97,
                block_size=8,workers=1,map_batch_size=4,probe_blocks=2)
            provenance=training_provenance(root/'run/checkpoint-1',manifest)
            self.assertEqual(provenance['status'],'verified')
            self.assertEqual(provenance['consumed_input_tokens'],16)
            self.assertEqual(provenance['consumed_scored_targets'],14)

    def test_all_arm_probe_additivity_batch_invariance_and_no_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); tok=fixture(root)
            data=preprocess_text(load_text_data(root/'eval'),tok,block_size=8,num_proc=1,batch_size=4)
            counts=count_tokens(data,97); mapping=frequency_buckets(counts,96,96)
            for arm in ARMS:
                set_seed(42); model=build_model(experiment_config(arm,tiny=True)).train()
                before={n:p.detach().clone() for n,p in model.named_parameters()}
                first=probe_gradients(model,{'en':data},{'en':[0,2,4]},mapping,batch_size=1)
                second=probe_gradients(model,{'en':data},{'en':[0,2,4]},mapping,batch_size=2)
                self.assertEqual(first['scored_targets'],21)
                self.assertEqual(first['optimizer_steps'],0)
                self.assertTrue(model.training)
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                for name,p in model.named_parameters(): torch.testing.assert_close(p,before[name],rtol=0,atol=0)
                for name,row in first['parameters'].items():
                    self.assertAlmostEqual(row['grad_rms'],second['parameters'][name]['grad_rms'],places=6)
                if arm=='B0':
                    self.assertLess(first['tied_paths']['overall']['additivity_relative_l2'],5e-5)
                elif arm in SHARED_ARMS:
                    self.assertEqual(first['tied_paths']['status'],
                                     'not_applicable_projected_or_partial_tying')
                else:
                    self.assertEqual(first['tied_paths']['status'],'not_applicable_untied')

    def test_full_checkpoint_cli_and_frequency_vs_ppl(self):
        from eval.ppl import evaluate as ppl
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); tok=fixture(root)
            prepare(root/'train',root/'eval',root/'tokenizer',root/'bundle',vocab_size=97,
                block_size=8,workers=1,map_batch_size=4,probe_blocks=2)
            manifest,counts,mapping,data=load_bundle(root/'bundle')
            reports=[]
            for arm in ('B0','C'):
                model=build_model(experiment_config(arm,tiny=True))
                model.save_pretrained(root/arm); tok.save_pretrained(root/arm)
                output=root/f'{arm}.json'
                argv=['diagnostics','--checkpoint',str(root/arm),'--diagnostic-bundle',str(root/'bundle'),
                    '--device','cpu','--precision','fp32','--output',str(output)]
                with patch.object(sys,'argv',argv): diagnostic_main()
                report=json.loads(output.read_text()); reports.append(report)
                actual=report['diagnostics']['frequency']['by_language']['en']
                expected=ppl(model,tok,root/'eval',block_size=8,workers=1,map_batch_size=4)
                self.assertAlmostEqual(actual['overall']['nll'],expected['by_language']['en']['nll'],places=6)
                self.assertEqual(sum(r['scored_targets'] for r in actual['buckets'].values()),
                                 actual['overall']['scored_targets'])
                for batch_size in (1,3):
                    result=frequency_nll(model,data,mapping,batch_size=batch_size)
                    self.assertAlmostEqual(result['token_weighted']['overall']['nll'],actual['overall']['nll'],places=6)
            self.assertIn('head',compare(*reports)['by_language']['en'])
            reports[1]['diagnostic_manifest_sha256']='other'
            with self.assertRaises(ValueError): compare(*reports)

    def test_bf16_diagnostics_all_arms_and_multilingual_aggregation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); tok=fixture(root)
            data=preprocess_text(load_text_data(root/'eval'),tok,block_size=8,num_proc=1,batch_size=4).select([0,1])
            counts=count_tokens(data,97); mapping=frequency_buckets(counts,96,96)
            for arm in ARMS:
                model=build_model(experiment_config(arm,tiny=True)).eval()
                observed=[]
                hook=model.model.layers[0].self_attn.q_proj.register_forward_hook(lambda m,i,o: observed.append(o.dtype))
                result=frequency_nll(model,{'en':data,'vi':data},mapping,precision='bf16')
                probe=probe_gradients(model,{'en':data},{'en':[0,1]},mapping,precision='bf16')
                hook.remove()
                self.assertEqual(set(observed),{torch.bfloat16})
                self.assertEqual(result['token_weighted']['overall']['scored_targets'],28)
                self.assertAlmostEqual(result['token_weighted']['overall']['nll'],
                    result['by_language']['en']['overall']['nll'])
                self.assertEqual(probe['scored_targets'],14)

    def test_optional_training_observer_preclip_and_update_no_effect(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); tok=fixture(root)
            data=preprocess_text(load_text_data(root/'train'),tok,block_size=8,num_proc=1,batch_size=4).select(range(2))
            models=[]
            for instrument in (False,True):
                set_seed(42); model=build_model(experiment_config('B0',tiny=True))
                args=TrainingArguments(output_dir=str(root/str(instrument)),use_cpu=True,
                    per_device_train_batch_size=1,gradient_accumulation_steps=2,num_train_epochs=1,
                    learning_rate=.001,max_grad_norm=1e-6,save_strategy='no',report_to='none',disable_tqdm=True)
                trainer=CausalTrainer(model=model,args=args,train_dataset=data,data_collator=default_data_collator)
                if instrument: attach_training_diagnostics(trainer,root/'stats',every=1,update_every=1)
                trainer.train(); models.append(model)
            for a,b in zip(models[0].parameters(),models[1].parameters()): torch.testing.assert_close(a,b,rtol=0,atol=0)
            record=json.loads((root/'stats/step-1.json').read_text())
            self.assertIn('pre_clipping',record['gradient_stage'])
            self.assertGreater(max(r['grad_l2'] for r in record['parameters'].values()),1e-6)
            self.assertGreater(max(r['update_rms'] for r in record['updates'].values()),0)

    def test_queue_includes_diagnostics_and_checks_result_manifest(self):
        from eval.eval_parallel import build_jobs
        from eval.parallel import run
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'checkpoint').mkdir(); (root/'bundle').mkdir()
            (root/'checkpoint/config.json').write_text('{}'); (root/'bundle/manifest.json').write_text('{}')
            args=SimpleNamespace(checkpoints=[f'B0={root}/checkpoint'],languages='en',output_dir=str(root/'out'),
                dataset_root=str(root/'bench'),eval_dir=str(root/'eval'),precision='bf16',benchmark_batch_size=8,
                tokenizer_name=None,task_groups=['hellaswag'],batch_size=1,preprocessing_num_workers=1,
                preprocessing_batch_size=4,preprocessing_cache_dir=None,diagnostic_bundle=str(root/'bundle'),
                diagnostics=['frequency','spectra','gradients'])
            jobs=build_jobs(args)
            self.assertEqual(len(jobs),5)
            self.assertEqual([j['tasks'] for j in jobs[2:]],[['frequency'],['spectra'],['gradients']])
            for valid in (False,True):
                out=root/f'queue_{valid}'; result=out/'result.json'
                payload=dict(success=True,checkpoint={'path':str(root/'checkpoint')},languages=['en'],
                    diagnostic_manifest_sha256='right' if valid else 'wrong',diagnostics={'frequency':{}})
                command=[sys.executable,'-c','import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2])',str(result),json.dumps(payload)]
                job=dict(name='probe',argv=command,result=str(result),checkpoint=str(root/'checkpoint'),
                    stage='diagnostics',tasks=['frequency'],expected=dict(languages=['en'],diagnostic_manifest_sha256='right'))
                with patch('scripts.gpu_status.require_free'):
                    if valid: run([job],[0],out,root)
                    else:
                        with self.assertRaises(RuntimeError): run([job],[0],out,root)
                self.assertEqual((out/'complete.json').exists(),valid)


if __name__=='__main__': unittest.main()
