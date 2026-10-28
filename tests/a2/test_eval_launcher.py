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

class FailedEvalHandoffTests(unittest.TestCase):
    def test_exact_reviewed_failure_cleanup_and_reject_wrong_gate(self):
        from scripts.launch_a2_downstream import verify_failed_eval_handoff
        from eval.a2_run import write,lock
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d);root=parent/'failed';burn_script=root/'source/resources/llm_pretrain_burn.py'
            records=dict(run=dict(status='failed',jobs=[dict(name='a2-downstream',returncode=1)]),
                supervisor=dict(training_status='failed',finished_at='fixture',free_after_training=[dict(pids=[])]*8),
                burn=dict(collective_progress_verified=True),
                gate=dict(passed=False,production_batch_size=8,manual_max_nats_per_token=1e-7,
                    batch_max_nats_per_token=0.,production_max_nats_per_token=.0359,
                    greedy_reference_mismatches=0,greedy_batch_mismatches=0))
            paths=dict(run='supervised/run/run.json',supervisor='supervised/supervisor.json',
                burn='supervised/burn-verified.json',gate='supervised/run/downstream/dclm-P300-original-check.json')
            reviewed=dict(scope='a2_batch8_numerical_gate_failure_before_scoring')
            for name,path in paths.items():reviewed[name]=write(root/path,records[name])
            rl=write(parent/'reviewed.json',reviewed)
            self.assertEqual(verify_failed_eval_handoff(parent,burn_script,rl),reviewed)
            gate_path=Path(reviewed['gate']['path']);records['gate']['manual_max_nats_per_token']=.1
            gate_path.write_text(json.dumps(records['gate']));reviewed['gate']=lock(gate_path)
            Path(rl['path']).write_text(json.dumps(reviewed));rl=lock(rl['path'])
            with self.assertRaises(ValueError):verify_failed_eval_handoff(parent,burn_script,rl)
