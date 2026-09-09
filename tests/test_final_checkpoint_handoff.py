"""No CUDA queries or remote actions: final-only plan and validator checks."""
import ast
import json
from pathlib import Path
import re
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from scripts.final_checkpoint_diagnostics import ARMS, diagnostic_jobs, verify
from finetune.run_all import build_jobs


class FinalCheckpointTests(unittest.TestCase):
    def test_final_only_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm in ARMS:
                p = root/arm/'checkpoint-5000'
                p.mkdir(parents=True)
                (p/'config.json').write_text('{}')
            bundle = root/'bundle'
            bundle.mkdir()
            (bundle/'manifest.json').write_text('{}')
            jobs = diagnostic_jobs(root, bundle, root/'diagnostics', root/'tokenizer')
            self.assertEqual(len(jobs), 18)
            self.assertEqual({j['stage'] for j in jobs}, {'diagnostics'})
            self.assertEqual({j['tasks'][0] for j in jobs}, {'frequency', 'spectra', 'gradients'})
            args = SimpleNamespace(checkpoints=[f'{arm}_step5000={root/arm/"checkpoint-5000"}' for arm in ARMS],
                languages='en', train_language='en', seeds=[42,123,456],
                tasks=['hellaswag','arc_easy','xnli'], output_dir=str(root/'finetune'),
                dataset_root=str(root/'benchmarks'), tokenizer_name=str(root/'tokenizer'),
                precision='bf16', benchmark_batch_size=8)
            ft = build_jobs(args)
            self.assertEqual(len(ft), 54)
            self.assertEqual(len({j['name'] for j in ft}), 54)
            self.assertTrue(all(Path(j['checkpoint']).name == 'checkpoint-5000' for j in jobs+ft))

    def test_reject_incomplete_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'complete.json').write_text(json.dumps(dict(success=True, completed=[])))
            for stage in ('diagnostics', 'finetune'):
                with self.assertRaises(AssertionError):
                    verify(stage, root, {})

    def test_shell_order_and_syntax(self):
        path = Path(__file__).resolve().parents[1]/'scripts/final_checkpoint_diagnostics_finetune_b200.sh'
        subprocess.run(['bash','-n',str(path)], check=True)
        source = path.read_text()
        for block in re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.S):
            ast.parse(block)
        self.assertLess(source.index('scripts.smoke_final_diagnostics'), source.index('"${TASK_DIAG[@]}"\n'))
        self.assertLess(source.index('"${TASK_DIAG[@]}"\n'), source.index('"${TASK_FT[@]}"\n'))
        self.assertLess(source.index('"${TASK_DIAG[@]}" --verify-finetune'), source.index("/'complete.json', dict(success=True"))


if __name__ == '__main__':
    unittest.main()
