import json
from pathlib import Path
import tempfile
import unittest

from scripts.launch_a2_downstream import freeze_source, job_config, remaining_budget
from run_experiments import load_jobs
from efficient_llm.diagnostic_data import sha256


class EvalLauncherTests(unittest.TestCase):
    def test_frozen_source_includes_native_eval_and_unchanged_supervisor(self):
        source=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);frozen=freeze_source(source,root,'a'*40)
            manifest=json.loads((root/'source-manifest.json').read_text())
            for name in ('eval/a2_run.py','eval/a2_score.py','eval/a2_requests.py',
                         'eval/benchmark_data.py','scripts/train_then_burn.py','run_experiments.py',
                         'resources/accelerate_b200.yaml','resources/llm_pretrain_burn.py'):
                self.assertEqual(manifest['files'][name],sha256(source/name))
                self.assertEqual(sha256(frozen/name),sha256(source/name))
            self.assertFalse(any('temp/' in name for name in manifest['files']))

    def test_native_runner_accepts_bounded_screen_contract(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'jobs.json';path.write_text(json.dumps(job_config({'path':'/plan.json','sha256':'a'*64},10800)))
            jobs=load_jobs(path);self.assertEqual(len(jobs),1)
            self.assertEqual(jobs[0]['gpus'],list(range(8)))
            self.assertEqual(jobs[0]['timeout_seconds'],10860)
            self.assertEqual(jobs[0]['required_outputs'][0]['json_equals'],dict(status='ok',models=14))

    def test_reserve_cleanup_within_phase_cap(self):
        self.assertFalse(remaining_budget(11*3600,120,3*3600)['report_due'])
        self.assertTrue(remaining_budget(13*3600,120,3*3600)['report_due'])
        for args in ((17*3600,120,3600),(0,0,10801),(0,0,float('nan'))):
            with self.assertRaises(ValueError):remaining_budget(*args)


if __name__=='__main__':unittest.main()
