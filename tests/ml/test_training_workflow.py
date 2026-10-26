"""Run the shipped queue with real tiny local assets, then compare native resume."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import test_baseline as baseline

ROOT = Path(__file__).resolve().parents[2]


class TrainingWorkflowTests(unittest.TestCase):
    def test_real_train_resume_validate_evaluate_summary_queue(self):
        fixture = baseline.BaselineTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        validation = fixture.root / 'validation.parquet'
        baseline.Dataset.from_dict({'text': ['four three two one ' * 8] * 3}).to_parquet(validation)
        substitutions = {'/absolute/model/config.json': str(fixture.assets / 'config.json'),
            '/absolute/tokenizer': str(fixture.assets), '/absolute/data/train': str(fixture.data),
            '/absolute/data/validation': str(validation)}
        config = json.loads((ROOT / 'jobs.training.example.json').read_text())
        for job in config['jobs']:
            job['argv'] = [substitutions.get(arg, arg) for arg in job['argv']]
        manifest = fixture.root / 'jobs.json'
        manifest.write_text(json.dumps(config))
        output = fixture.root / 'queue'
        run = subprocess.run([sys.executable, str(ROOT / 'run_experiments.py'), '--config', str(manifest),
            '--project-dir', str(ROOT), '--run-dir', str(output)], capture_output=True, text=True,
            timeout=300, env=os.environ.copy())
        logs = '\n'.join(p.name + '\n' + p.read_text()[-8000:] for p in sorted(output.glob('*.log')))
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr + logs)
        completed = json.loads((output / 'complete.json').read_text())
        self.assertEqual(len(completed['jobs']), 6)
        self.assertTrue(all(j['status'] == 'ok' for j in completed['jobs']))
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual([row['step'] for row in summary['stages']], [2, 4])
        self.assertTrue(all(row['schedule_steps'] == 8 for row in summary['stages']))
        self.assertEqual(summary['evaluation']['step'], 4)
        self.assertGreater(summary['evaluation']['num_tokens'], 0)
        self.assertAlmostEqual(summary['evaluation']['loss'], summary['stages'][-1]['validation']['loss'], places=6)
        self.assertTrue((output / 'summary.md').is_file())
        # The runner/verification stages do not alter the native continuation.
        command = [x.replace('{python}', sys.executable).replace('{project_dir}', str(ROOT))
            .replace('{run_dir}', str(fixture.root / 'reference')) for x in config['jobs'][0]['argv']]
        command[command.index('--stop_after') + 1] = '4'
        reference = subprocess.run(command, text=True, capture_output=True, timeout=120, env=os.environ.copy())
        self.assertEqual(reference.returncode, 0, reference.stdout + reference.stderr)
        a = baseline.AutoModelForCausalLM.from_pretrained(output / 'training', local_files_only=True)
        b = baseline.AutoModelForCausalLM.from_pretrained(fixture.root / 'reference/training', local_files_only=True)
        for name, value in a.state_dict().items():
            baseline.torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)
        # A fresh-run requirement must preserve every existing result on retry.
        before = (output / 'summary.json').read_bytes()
        retry = subprocess.run([sys.executable, str(ROOT / 'run_experiments.py'), '--config', str(manifest),
            '--project-dir', str(ROOT), '--run-dir', str(output)], capture_output=True, text=True, timeout=20)
        self.assertNotEqual(retry.returncode, 0)
        self.assertEqual((output / 'summary.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
