"""CPU-only grid and launch-contract tests for the 14-arm result handoff."""
import ast
from pathlib import Path
import re
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from scripts.next_capacity_eval_handoff import ARMS, EVAL_STEPS, checkpoint_grid


class NextCapacityEvalHandoffTests(unittest.TestCase):
    def test_grid_is_exact_and_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            grid = checkpoint_grid(root)
        self.assertEqual(len(ARMS), 14)
        self.assertEqual(EVAL_STEPS, (250, 500, 1000, 2000, 3000, 4000, 5000))
        self.assertEqual(len(grid), 98)
        self.assertEqual(len({(arm, step) for arm, step, _ in grid}), 98)
        self.assertTrue(all(path == root/arm/f"checkpoint-{step}"
                            for arm, step, path in grid))

    def test_exact_evaluation_and_final_finetune_plans(self):
        from eval.benchmarks import DEFAULT_GROUPS
        from eval.eval_parallel import build_jobs as eval_jobs
        from finetune.run_all import build_jobs as finetune_jobs
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_specs = []
            final_specs = []
            for arm, step, checkpoint in checkpoint_grid(root):
                checkpoint.mkdir(parents=True)
                (checkpoint/"config.json").write_text("{}")
                checkpoint_specs.append(f"{arm}_step{step}={checkpoint}")
                if step == 5000:
                    final_specs.append(f"{arm}_step5000={checkpoint}")
            bundle = root/"bundle"
            bundle.mkdir()
            (bundle/"manifest.json").write_text("{}")
            common = dict(languages="en", dataset_root=str(root/"benchmarks"),
                tokenizer_name=str(root/"tokenizer"), precision="bf16",
                benchmark_batch_size=8, gpus=list(range(8)))
            evaluation = eval_jobs(SimpleNamespace(checkpoints=checkpoint_specs,
                output_dir=str(root/"eval"), eval_dir=str(root/"eval_data"),
                task_groups=list(DEFAULT_GROUPS), batch_size=1,
                preprocessing_num_workers=160, preprocessing_batch_size=1000,
                preprocessing_cache_dir=str(root/"cache"),
                diagnostic_bundle=str(bundle),
                diagnostics=["frequency", "spectra", "gradients"], **common))
            finetune = finetune_jobs(SimpleNamespace(checkpoints=final_specs,
                output_dir=str(root/"finetune"), train_language="en",
                tasks=["hellaswag", "arc_easy", "xnli"],
                seeds=[42, 123, 456], **common))
        self.assertEqual(len(evaluation), 490)
        self.assertEqual(sum(job["stage"] == "diagnostics" for job in evaluation), 294)
        self.assertEqual(len({job["name"] for job in evaluation}), 490)
        self.assertEqual(len(finetune), 126)
        self.assertEqual(len({job["name"] for job in finetune}), 126)
        self.assertTrue(all(Path(job["checkpoint"]).name == "checkpoint-5000"
                            for job in finetune))

    def test_shell_syntax_order_and_counts(self):
        path = Path(__file__).resolve().parents[1]/"scripts/eval_diagnostics_finetune_next_capacity_b200.sh"
        subprocess.run(["bash", "-n", str(path)], check=True)
        source = path.read_text()
        for block in re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.S):
            ast.parse(block)
        self.assertIn('test "${#TASK_EVAL_CHECKPOINTS[@]}" = 98', source)
        self.assertIn('test "${#TASK_FINAL_CHECKPOINTS[@]}" = 14', source)
        self.assertLess(source.index('echo START_98_CHECKPOINT_FULL_EVALUATION_AND_DIAGNOSTICS'),
                        source.index('echo START_14_FINAL_CHECKPOINT_ENGLISH_FINETUNING'))
        self.assertLess(source.index('verify-eval'),
                        source.index('echo START_14_FINAL_CHECKPOINT_ENGLISH_FINETUNING'))
        self.assertLess(source.index('verify-finetune'), source.index("/'complete.json'"))


if __name__ == "__main__":
    unittest.main()
