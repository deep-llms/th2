from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from efficient_llm.diagnostic_data import sha256
from scripts.launch_a2_downstream import freeze_source
from scripts.launch_a2_stage1_downstream import budget, job_config, bind_predecessor
from run_experiments import load_jobs


class Stage1EvalLauncherTest(unittest.TestCase):
    def test_stage_budget_and_absolute_deadline_include_cleanup(self):
        now=datetime(2026,10,29,tzinfo=timezone.utc)
        deadline=(now+timedelta(hours=10)).isoformat()
        self.assertFalse(budget(15*3600,60,7200,deadline,now)['report_due'])
        self.assertTrue(budget(29*3600,60,7200,deadline,now)['report_due'])
        for values in ((35*3600,0,3600,deadline),(0,0,float('nan'),deadline),
                       (0,0,10801,deadline),(0,0,7200,(now+timedelta(hours=2)).isoformat()),
                       (0,0,1,'2026-10-30T00:00:00')):
            with self.assertRaises(ValueError):budget(*values,now=now)

    def test_native_supervisor_contract_requires_six_models_and_paired_report(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'jobs.json';path.write_text(json.dumps(job_config(dict(path='/plan',sha256='a'*64),7200)))
            jobs=load_jobs(path)
            self.assertEqual(len(jobs),1);job=jobs[0]
            self.assertEqual(job['name'],'stage1-downstream')
            self.assertEqual(job['gpus'],list(range(8)))
            self.assertEqual(job['timeout_seconds'],7260)
            self.assertIn('eval.a2_stage1_run',job['argv'])
            self.assertEqual(job['required_outputs'][0]['json_equals'],dict(status='ok',models=6))
            self.assertEqual(job['required_outputs'][-1]['json_equals'],dict(resamples=10000,seed=0))

    def test_freeze_includes_stage_spec_and_original_supervisor_bytes(self):
        source=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);frozen=freeze_source(source,root,'a'*40,extra_files=('docs/A2_stage1_spec.md',))
            manifest=json.loads((root/'source-manifest.json').read_text())
            for name in ('docs/A2_stage1_spec.md','eval/a2_stage1_run.py','eval/a2_stage1_plan.py',
                         'scripts/launch_a2_stage1_downstream.py','scripts/train_then_burn.py'):
                self.assertEqual(manifest['files'][name],sha256(source/name))
                self.assertEqual(sha256(frozen/name),sha256(source/name))

    def test_predecessor_paths_must_match_reviewed_training(self):
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d);prior=parent/'training';burn=prior/'source/resources/llm_pretrain_burn.py'
            paths={'supervisor':'supervised/supervisor.json','run_complete':'supervised/run/complete.json',
                   'burn':'supervised/burn-verified.json'}
            for key,name in [('results','runs.json'),('choice','lr_choice.json'),('tests','tests.json'),
                             ('decision','decision_stage1.json'),('complete','complete.json')]:
                paths[key]='supervised/run/results/a2_stage1/'+name
            stage={k:dict(path=str(prior/v),sha256='a'*64) for k,v in paths.items()}
            with patch('scripts.launch_a2_stage1_downstream.validate_terminal_bundle',return_value=(stage,None,None,None)), \
                 patch('scripts.launch_a2_stage1_downstream.verify_handoff') as handoff:
                self.assertEqual(bind_predecessor({}, {},parent,burn),prior)
                handoff.assert_called_once()
                stage['complete']['path']=str(parent/'another/complete.json')
                handoff.reset_mock()
                with self.assertRaises(ValueError):bind_predecessor({}, {},parent,burn)
                handoff.assert_not_called()


if __name__=='__main__':unittest.main()
