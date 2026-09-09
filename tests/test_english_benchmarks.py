"""Offline fixtures exercise the actual pinned harness, including every BLiMP task."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from datasets import Dataset
import torch
import yaml

from capacity_allocation.modeling import build_model, experiment_config
from eval.benchmarks import ENGLISH_CORE_GROUPS, load_tasks, task_plan
from eval.blimp_tasks import BLIMP_TASKS
from test_capacity_data_train import fixture


def benchmark_fixtures(root):
    """Tiny synthetic rows with official dataset schemas; not benchmark results."""
    configs = {}

    def add(repo, config, split, rows):
        directory = root/repo
        path = directory/config/f'{split}.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        Dataset.from_dict(rows).to_parquet(path)
        configs.setdefault(repo, {})[config] = dict(config_name=config,
            data_files=[dict(split=split, path=f'{config}/{split}.parquet')])

    for task in BLIMP_TASKS:
        add('nyu-mll/blimp', task.removeprefix('blimp_'), 'train',
            dict(sentence_good=['w1 w2 w3', 'w4 w5 w6'], sentence_bad=['w3 w2 w1', 'w6 w5 w4']))
    add('EleutherAI/lambada_openai', 'default', 'test', dict(text=['w1 w2 w3', 'w4 w5 w6']))
    add('baber/piqa', 'default', 'validation', dict(goal=['w1', 'w2'],
        sol1=['w3', 'w4'], sol2=['w5', 'w6'], label=[0,1]))
    add('allenai/winogrande', 'winogrande_xl', 'validation',
        dict(sentence=['w1 _ w2', 'w3 _ w4'], option1=['w5', 'w6'], option2=['w7', 'w8'], answer=['1','2']))
    for config in ('ARC-Easy', 'ARC-Challenge'):
        add('allenai/ai2_arc', config, 'test', dict(question=['w1', 'w2'],
            choices=[{'label':['A','B','C','D'], 'text':['w3','w4','w5','w6']}]*2, answerKey=['A','B']))
        # Official ARC exposes train/validation/test; ConfigurableTask reads
        # its training split for few-shot setup even with zero few-shot examples.
        for split in ('train', 'validation'):
            directory = root/'allenai/ai2_arc'
            source = Dataset.from_parquet(str(directory/config/'test.parquet'))
            source.to_parquet(directory/config/f'{split}.parquet')
            configs['allenai/ai2_arc'][config]['data_files'].append(
                dict(split=split, path=f'{config}/{split}.parquet'))
    add('aps/super_glue', 'boolq', 'validation', dict(passage=['w1 w2', 'w3 w4'], question=['w5', 'w6'], label=[0,1]))
    add('Rowan/hellaswag', 'default', 'validation', dict(ctx_a=['w1 w2']*2, ctx_b=['w3']*2,
        activity_label=['w4']*2, endings=[['w5','w6','w7','w8']]*2, label=['0','1']))
    for repo in ('baber/piqa','allenai/winogrande','aps/super_glue','Rowan/hellaswag'):
        for config, row in configs[repo].items():
            directory = root/repo
            source = Dataset.from_parquet(str(directory/config/'validation.parquet'))
            source.to_parquet(directory/config/'train.parquet')
            row['data_files'].append(dict(split='train', path=f'{config}/train.parquet'))
    for repo, rows in configs.items():
        (root/repo/'README.md').write_text('---\n'+yaml.safe_dump({'configs':list(rows.values())})+'---\n')


@unittest.skipUnless(importlib.util.find_spec('lm_eval'), 'Requires lm_eval==0.4.10')
class EnglishBenchmarksTests(unittest.TestCase):
    def test_all_core_tasks_through_actual_eval_cli(self):
        from eval.eval_checkpoint import main
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tokenizer = fixture(root)
            benchmark_fixtures(root/'benchmarks')
            plan, missing = task_plan('en', ENGLISH_CORE_GROUPS)
            self.assertFalse(missing)
            self.assertEqual(len(plan), 74)
            tasks = load_tasks(plan, root/'benchmarks')
            self.assertEqual(tasks['winogrande'].multiple_input, 2)
            self.assertEqual(tasks['lambada_openai'].config.output_type, 'loglikelihood')
            for arm in ('B0', 'C'):
                checkpoint = root/arm
                build_model(experiment_config(arm, tiny=True)).save_pretrained(checkpoint)
                tokenizer.save_pretrained(checkpoint)
                output = root/f'{arm}_eval.json'
                argv = ['eval.eval_checkpoint', '--checkpoint', str(checkpoint), '--bench-only',
                    '--dataset-root', str(root/'benchmarks'), '--task-groups', *ENGLISH_CORE_GROUPS,
                    '--device', 'cpu', '--precision', 'fp32', '--benchmark-batch-size', '4', '--output', str(output)]
                with patch.object(sys, 'argv', argv):
                    main()
                result = json.loads(output.read_text())
                self.assertTrue(result['success'])
                self.assertEqual(set(result['benchmarks']), {p['task'] for p in plan})
                self.assertTrue(all(r['samples']['effective'] == 2 for r in result['benchmarks'].values()))
                self.assertGreater(result['benchmarks']['lambada_openai']['metrics']['perplexity,none'], 0)
                self.assertEqual(result['benchmark_summaries']['blimp']['subtasks'], 67)
                self.assertEqual(result['benchmark_summaries']['blimp']['samples'], 134)


if __name__ == '__main__':
    unittest.main()
