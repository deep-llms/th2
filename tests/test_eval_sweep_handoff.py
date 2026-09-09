"""CPU-only checks for the fixed six-arm/seven-checkpoint launch plan."""
import ast
from collections import Counter
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from eval.benchmarks import DEFAULT_GROUPS
from eval.eval_parallel import build_jobs as eval_jobs
from finetune.run_all import build_jobs as finetune_jobs


class SweepHandoffTests(unittest.TestCase):
    def test_complete_checkpoint_task_seed_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = []
            for arm in ('B0', 'A128', 'A256', 'A512', 'C', 'D'):
                for step in (250, 500, 1000, 2000, 3000, 4000, 5000):
                    path = root/arm/f'checkpoint-{step}'
                    path.mkdir(parents=True)
                    (path/'config.json').write_text(json.dumps({}))
                    checkpoints.append(f'{arm}_step{step}={path}')
            args = SimpleNamespace(checkpoints=checkpoints, languages='en',
                tokenizer_name=str(root/'tokenizer'), dataset_root=str(root/'benchmarks'),
                precision='bf16', benchmark_batch_size=8, output_dir=str(root/'eval'),
                task_groups=list(DEFAULT_GROUPS), eval_dir=str(root/'data'), batch_size=1,
                preprocessing_num_workers=160, preprocessing_batch_size=1000,
                preprocessing_cache_dir=str(root/'cache'), train_language='en',
                seeds=[42, 123, 456], tasks=['hellaswag', 'arc_easy', 'xnli'])
            jobs = eval_jobs(args)
            self.assertEqual(len(jobs), 84)
            self.assertEqual(Counter(j['stage'] for j in jobs), dict(ppl=42, benchmarks=42))
            self.assertTrue(all(len(j['tasks']) == 78 for j in jobs if j['stage'] == 'benchmarks'))
            args.output_dir = str(root/'finetune')
            jobs = finetune_jobs(args)
            self.assertEqual(len(jobs), 378)
            self.assertEqual(len({j['name'] for j in jobs}), 378)
            self.assertEqual(len({j['result'] for j in jobs}), 378)
            self.assertEqual(set(Counter(j['checkpoint'] for j in jobs).values()), {9})
            self.assertEqual(Counter(j['expected']['task'] for j in jobs),
                             dict(hellaswag=126, arc_easy=126, xnli=126))
            for job in jobs:
                self.assertEqual(job['expected']['languages'], ['en'])
                self.assertIn(job['expected']['seed'], [42, 123, 456])
                self.assertTrue(job['checkpoint'].startswith(str(root)))
                self.assertNotIn('/finetune/', job['checkpoint'])

    def test_shell_and_embedded_python_syntax(self):
        script = Path(__file__).resolve().parents[1]/'scripts/eval_finetune_capacity_b200.sh'
        subprocess.run(['bash', '-n', str(script)], check=True)
        source = script.read_text()
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.S)
        self.assertEqual(len(blocks), 5)
        for block in blocks:
            ast.parse(block)
        self.assertIn('set -euo pipefail', source)
        self.assertLess(source.index('verify_stage eval 84'), source.index('"${TASK_FINETUNE[@]}"\n'))
        self.assertLess(source.index('verify_stage finetune 378'), source.index("/'complete.json', dict(success=True"))

    def test_stage_gate_rejects_changed_identity_or_partial_ppl(self):
        script = Path(__file__).resolve().parents[1]/'scripts/eval_finetune_capacity_b200.sh'
        block = re.findall(r"<<'PY'\n(.*?)\nPY", script.read_text(), flags=re.S)[3]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'eval').mkdir()
            identity = dict(path='/original/B0/checkpoint-250', files={'model.safetensors': 'abc'})
            reference = dict(checkpoints={'B0_step250': identity}, scored_targets=9829694,
                             ppl_fingerprint='packed', benchmark_coverage={})
            (root/'inputs_verified.json').write_text(json.dumps(reference))
            row = dict(scored_targets=9829694, data_fingerprint='packed')
            item = dict(success=True, languages=['en'], checkpoint=identity,
                        ppl=dict(by_language=dict(en=row)))
            result_path = root/'eval'/'ppl.json'
            complete = dict(success=True, completed=[dict(exit_code=0, error=None, result=str(result_path))])
            (root/'eval'/'complete.json').write_text(json.dumps(complete))

            def check():
                result_path.write_text(json.dumps(item))
                with patch.object(sys, 'argv', ['gate', str(root), 'eval', '1']), redirect_stdout(io.StringIO()):
                    exec(compile(block, 'verify_stage', 'exec'), {})

            check()
            row['scored_targets'] -= 1
            with self.assertRaises(AssertionError):
                check()
            row['scored_targets'] += 1
            item['checkpoint'] = dict(path=identity['path'], files={'model.safetensors': 'changed'})
            with self.assertRaises(AssertionError):
                check()


if __name__ == '__main__':
    unittest.main()
