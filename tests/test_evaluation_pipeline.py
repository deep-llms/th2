import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from datasets import Dataset
from capacity_allocation.modeling import ARMS, build_model, experiment_config
from eval.benchmarks import (DEFAULT_GROUPS, LEGACY_GROUPS, ENGLISH_CORE_GROUPS, TASKS,
                             local_task_config, task_plan, summarize_benchmarks)
from eval.blimp_tasks import BLIMP_TASKS
from eval.runtime import checkpoint_specs, languages, load_checkpoint
from eval.ppl import evaluate as ppl_eval
from finetune.tasks import GenerativeDataset, encode_example, format_example
from finetune.train import fit
from test_capacity_data_train import fixture


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_english_default_tasks_and_multilingual_selection(self):
        plan, missing = task_plan()
        self.assertEqual([p['task'] for p in plan[:6]], ['xnli_en', 'belebele_eng_Latn',
                         'xstorycloze_en', 'paws_en', 'hellaswag', 'arc_easy'])
        self.assertEqual(len(plan), 78)
        self.assertEqual(len({p['task'] for p in plan}), 78)
        self.assertEqual(sum(p['group'] == 'blimp' for p in plan), 67)
        self.assertEqual(len(task_plan('en', LEGACY_GROUPS)[0]), 6)
        self.assertEqual(len(task_plan('en', ENGLISH_CORE_GROUPS)[0]), 74)
        self.assertFalse(missing)
        plan, missing = task_plan('en,zh', list(DEFAULT_GROUPS)+['xcopa'])
        self.assertTrue(all(p['language'] in ('en', 'zh') for p in plan))
        self.assertIn(dict(group='hellaswag', language='zh'), missing)
        self.assertIn(dict(group='xcopa', language='en'), missing)
        with self.assertRaises(ValueError):
            task_plan('en,zh', ['hellaswag'])
        for selection in ('en,en', '../en', 'en,', 'unknown'):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                languages(selection)

    def test_blimp_complete_macro_mean_and_missing_subtest(self):
        results = {name: dict(metrics={'acc,none': (i % 2)}, samples={'effective': i+1})
                   for i, name in enumerate(BLIMP_TASKS)}
        summary = summarize_benchmarks(results)['blimp']
        self.assertEqual(summary['subtasks'], 67)
        self.assertAlmostEqual(summary['metrics']['acc,none'], 33/67)
        self.assertEqual(summary['samples'], sum(range(1,68)))
        self.assertEqual(summarize_benchmarks({'hellaswag': {}}), {})
        results.pop(BLIMP_TASKS[0])
        with self.assertRaisesRegex(ValueError, 'missing subtests'):
            summarize_benchmarks(results)

    def test_lambada_only_accepts_loglikelihood_definition(self):
        plan, _ = task_plan('en', ['lambada'])
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'EleutherAI/lambada_openai').mkdir(parents=True)
            config = dict(task='lambada_openai', output_type='loglikelihood',
                          dataset_path='EleutherAI/lambada_openai')
            self.assertEqual(local_task_config(config, plan[0], folder)['output_type'], 'loglikelihood')
            with self.assertRaises(ValueError):
                local_task_config(dict(config, output_type='multiple_choice'), plan[0], folder)

    def test_offline_config_only_requires_selected_snapshot(self):
        plan, _ = task_plan('en', ['hellaswag'])
        with tempfile.TemporaryDirectory() as folder:
            config = dict(task='hellaswag', output_type='multiple_choice', dataset_path='Rowan/hellaswag')
            with self.assertRaises(FileNotFoundError):
                local_task_config(config, plan[0], folder)
            (Path(folder)/'Rowan/hellaswag').mkdir(parents=True)
            resolved = local_task_config(config, plan[0], folder)
            self.assertTrue(Path(resolved['dataset_path']).is_absolute())
            self.assertEqual(config['dataset_path'], 'Rowan/hellaswag')  # no in-place mutation
            for bad in (dict(config, dataset_path='other/repo'),
                        dict(config, dataset_kwargs={'data_files': 'https://example.com/file'})):
                with self.assertRaises(ValueError):
                    local_task_config(bad, plan[0], folder)

    def test_ppl_separate_languages_and_weighted_aggregate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tokenizer = fixture(root)
            model = build_model(experiment_config('C', tiny=True)).eval()
            both = ppl_eval(model, tokenizer, root/'eval', 'en,vi', block_size=8,
                            workers=2, map_batch_size=4)
            en = ppl_eval(model, tokenizer, root/'eval', 'en', block_size=8,
                          workers=2, map_batch_size=4)
            self.assertEqual(set(both['by_language']), {'en', 'vi'})
            self.assertEqual(both['by_language']['en'], en['by_language']['en'])
            self.assertEqual(both['token_weighted']['scored_targets'], 2*en['token_weighted']['scored_targets'])
            self.assertAlmostEqual(both['token_weighted']['nll'], en['token_weighted']['nll'])
            with self.assertRaises(FileNotFoundError):
                ppl_eval(model, tokenizer, root/'eval', 'zh')

    def test_loading_every_arm_and_finetune_update(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tokenizer = fixture(root)
            for arm in ARMS:
                path = root/arm
                build_model(experiment_config(arm, tiny=True)).save_pretrained(path)
                tokenizer.save_pretrained(path)
                model, loaded_tokenizer = load_checkpoint(path)
                item = encode_example('w1 w2', ' w3 w4', loaded_tokenizer, 8)
                before = model.get_input_embeddings().weight.detach().clone() if arm == 'B0' else next(model.parameters()).detach().clone()
                report = fit(model, [item, item], dict(batch_size=1, epochs=1, lr=1e-3),
                             device='cpu', precision='bf16', seed=42)
                self.assertEqual(report['steps'], 2)
                after = model.get_input_embeddings().weight if arm == 'B0' else next(model.parameters())
                self.assertFalse(torch.equal(before, after))

    def test_completion_masking_prefix_and_truncation(self):
        with tempfile.TemporaryDirectory() as folder:
            tokenizer = fixture(Path(folder))
            item = encode_example('w1 w2', ' w3 [EOS]', tokenizer, 8)
            self.assertEqual(item['labels'][:2].tolist(), [-100, -100])
            self.assertEqual(item['labels'][3].item(), tokenizer.eos_token_id)
            self.assertTrue(item['labels'][4:].eq(-100).all())
            empty = encode_example('', ' w3 w4', tokenizer, 8)
            self.assertEqual(empty['input_ids'][0].item(), tokenizer.eos_token_id)
            self.assertEqual(empty['labels'][0].item(), -100)
            self.assertEqual(empty['labels'][1].item(), tokenizer('w3')['input_ids'][0])
            self.assertIsNone(encode_example('w1 w2 w3', ' w4', tokenizer, 3))

    def test_dataset_rejects_no_training_or_no_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            tokenizer = fixture(Path(folder))
            task = SimpleNamespace(has_training_docs=lambda: False)
            with self.assertRaisesRegex(ValueError, 'never substitute'):
                GenerativeDataset(task, tokenizer)

    def test_job_plans_english_defaults_and_explicit_languages(self):
        from eval.eval_parallel import build_jobs as eval_jobs
        from finetune.run_all import build_jobs as ft_jobs
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'checkpoint').mkdir()
            (root/'checkpoint/config.json').write_text('{}')
            args = SimpleNamespace(checkpoints=[f'B0={root}/checkpoint'], languages='en',
                output_dir=str(root/'out'), dataset_root=str(root/'datasets'), eval_dir=str(root/'eval'),
                precision='bf16', benchmark_batch_size=8, tokenizer_name=None,
                task_groups=list(DEFAULT_GROUPS), batch_size=1, preprocessing_num_workers=160,
                preprocessing_batch_size=1000, preprocessing_cache_dir=None,
                train_language='en', seeds=[42, 123, 456], tasks=['hellaswag', 'arc_easy', 'xnli'])
            jobs = eval_jobs(args)
            self.assertEqual(len(jobs), 2)
            self.assertEqual([j['stage'] for j in jobs], ['ppl', 'benchmarks'])
            for job in jobs:
                self.assertEqual(job['expected']['languages'], ['en'])
                self.assertEqual(job['argv'][job['argv'].index('--languages')+1], 'en')
            jobs = ft_jobs(args)
            self.assertEqual(len(jobs), 9)
            self.assertEqual(len({j['result'] for j in jobs}), 9)
            args.languages = 'en,vi'
            jobs = ft_jobs(args)
            self.assertTrue(all(j['expected']['languages'] == ['en', 'vi'] for j in jobs))
            args.languages = 'en,zh'
            with self.assertRaises(ValueError):  # No fabricated Chinese HellaSwag.
                ft_jobs(args)
            args.checkpoints *= 2
            with self.assertRaises(ValueError):
                checkpoint_specs(args.checkpoints)

    def test_pool_result_validation_and_failure_propagation(self):
        from eval.parallel import run
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for failure in (False, True):
                out = root/str(failure)
                result = out/'result.json'
                data = dict(success=not failure, languages=['en'], checkpoint={'path': '/fixture'},
                            ppl={'by_language': {'en': {'nll': 1.}}})
                command = [sys.executable, '-c',
                    'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(sys.argv[2])',
                    str(result), json.dumps(data)]
                jobs = [dict(name='fixture', argv=command, result=str(result), checkpoint='/fixture',
                             stage='ppl', expected={'languages': ['en']}, tasks=[])]
                with patch('scripts.gpu_status.require_free') as check:
                    if failure:
                        with self.assertRaises(RuntimeError):
                            run(jobs, [3], out, root)
                        self.assertFalse((out/'complete.json').exists())
                    else:
                        run(jobs, [3], out, root)
                        self.assertTrue((out/'complete.json').exists())
                    self.assertGreaterEqual(check.call_count, 2)


@unittest.skipUnless(importlib.util.find_spec('lm_eval'), 'Requires optional lm_eval==0.4.10 test environment')
class HarnessIntegrationTests(unittest.TestCase):
    def test_actual_harness_forward_precision_all_arms(self):
        from scripts.check_eval_precision import check
        report = check('cpu')
        self.assertTrue(report['success'], report)
        self.assertEqual(len(report['checks']), 2 * len(ARMS))

    def test_exact_blimp_suite_matches_pinned_harness(self):
        from lm_eval.tasks import TaskManager
        manager = TaskManager()
        self.assertEqual(tuple(manager._get_config('blimp')['task']), BLIMP_TASKS)

    def test_bpe_completion_boundary_matches_harness(self):
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers
        from transformers import PreTrainedTokenizerFast
        from lm_eval.models.huggingface import HFLM
        backend = Tokenizer(models.BPE(unk_token='[UNK]'))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.train_from_iterator(['hello world', 'Question: hello Answer: world',
                                     'helloworld', 'another example']*10,
            trainers.BpeTrainer(vocab_size=80, special_tokens=['[UNK]', '[EOS]']))
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,
            unk_token='[UNK]', eos_token='[EOS]', pad_token='[EOS]')
        model = build_model(experiment_config('B0', tiny=True))
        lm = HFLM(pretrained=model, tokenizer=tokenizer, device='cpu', batch_size=1,
                  add_bos_token=False, prefix_token_id=tokenizer.eos_token_id)
        for prompt, completion in [('hello', 'world'), ('hello ', 'world'),
                                   ('Question: hello Answer:', ' world')]:
            context, continuation = lm._encode_pair(prompt, completion)
            item = encode_example(prompt, completion, tokenizer, 64)
            if not continuation:
                self.assertIsNone(item)
                continue
            self.assertEqual(item['input_ids'][:len(context)+len(continuation)].tolist(), context+continuation)
            self.assertEqual(item['labels'][:len(context)+len(continuation)].tolist(),
                             [-100]*len(context)+continuation)

    def test_all_selected_official_configs_accept_local_mapping(self):
        from lm_eval.tasks import TaskManager
        manager = TaskManager()
        plan, _ = task_plan('en,ar,de,ru,vi,zh', list(TASKS))
        with tempfile.TemporaryDirectory() as folder:
            for item in plan:
                (Path(folder)/item['repository']).mkdir(parents=True, exist_ok=True)
                config = local_task_config(manager._get_config(item['task']), item, folder)
                self.assertEqual(config['task'], item['task'])

    def test_real_local_dataset_harness_and_finetune_cli(self):
        from eval.benchmarks import load_tasks, evaluate
        from finetune.train import main as ft_main
        from eval.eval_checkpoint import main as eval_main
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tokenizer = fixture(root)
            data = root/'benchmarks/Rowan/hellaswag'
            data.mkdir(parents=True)
            rows = dict(ctx_a=['w1 w2']*4, ctx_b=['w3']*4, activity_label=['w4']*4,
                        endings=[['w5', 'w6', 'w7', 'w8']]*4, label=['0','1','2','3'])
            for split in ('train', 'validation'):
                Dataset.from_dict(rows).to_parquet(data/f'{split}.parquet')
            plan, _ = task_plan('en', ['hellaswag'])
            tasks = load_tasks(plan, root/'benchmarks')
            prompt, completion = format_example(tasks['hellaswag'], tasks['hellaswag'].training_docs()[0])
            self.assertEqual(prompt, 'w4: w1 w2 W3')
            self.assertEqual(completion, ' w5')
            for arm in ARMS:
                model = build_model(experiment_config(arm, tiny=True)).eval()
                report = evaluate(model, tokenizer, tasks, device='cpu', precision='fp32', batch_size=2)
                self.assertEqual(report['hellaswag']['samples']['effective'], 4)
            checkpoint = root/'C'
            build_model(experiment_config('C', tiny=True)).save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            argv = ['finetune.train', '--checkpoint', str(checkpoint), '--task', 'hellaswag',
                    '--dataset-root', str(root/'benchmarks'), '--device', 'cpu', '--precision', 'fp32',
                    '--num-workers', '0', '--output-dir', str(root/'ft')]
            with patch.object(sys, 'argv', argv):
                ft_main()
            result = json.loads((root/'ft/result.json').read_text())
            self.assertTrue(result['success'])
            self.assertEqual(result['languages'], ['en'])
            self.assertEqual(result['training']['steps'], 3)
            self.assertTrue((root/'ft/model/config.json').exists())
            argv = ['eval.eval_checkpoint', '--checkpoint', str(checkpoint),
                    '--dataset-root', str(root/'benchmarks'), '--eval-dir', str(root/'eval'),
                    '--task-groups', 'hellaswag', '--device', 'cpu', '--precision', 'fp32',
                    '--block-size', '8', '--preprocessing-num-workers', '2',
                    '--preprocessing-batch-size', '4', '--output', str(root/'eval.json')]
            with patch.object(sys, 'argv', argv):
                eval_main()
            result = json.loads((root/'eval.json').read_text())
            self.assertEqual(set(result['ppl']['by_language']), {'en'})
            self.assertEqual(set(result['benchmarks']), {'hellaswag'})


if __name__ == '__main__':
    unittest.main()
