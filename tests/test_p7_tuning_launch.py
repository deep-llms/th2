import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.p7_tuning_launch import OLD_NAME, PARTIAL, cleanup_target, cleanup
from scripts.p7_tuning_queue import ALL_CASES, CASES, make


class TuningLaunchTests(unittest.TestCase):
    def fixture(self, root):
        source=root/OLD_NAME;target=source/PARTIAL;target.mkdir(parents=True)
        def save(p,v):p.write_text(json.dumps(v))
        burn=dict(collective_progress_verified=True,session='burn')
        save(source/'supervised/burn-verified.json',burn)
        save(source/'supervised/supervisor.json',dict(training_status='failed',finished_at='now',burn=burn))
        save(source/'user-stop-after-keyonly.json',dict(status='stop_requested',burn_handoff=burn))
        save(target/'train_config.json',dict(pilot=dict(arm='P7-simple-lm-noaux',proxy_lambda_max=0),
            training=dict(seed=42,data_seed=42,max_steps=2500)))
        (target/'checkpoint-250').mkdir();(target/'checkpoint-250/model.safetensors').write_bytes(b'test')
        return source,target

    def test_cleanup_preserves_neighbors_and_refuses_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();source,target=self.fixture(root)
            sibling=source/'supervised/run/lm-valueonly';sibling.mkdir();(sibling/'keep').write_text('preserve')
            run=root/'fresh';run.mkdir();(run/'smoke-recipes.json').write_text('{"status":"passed"}')
            self.assertEqual(cleanup_target(source),target)
            (target/'result.json').write_text('{}')
            with self.assertRaises(ValueError):cleanup_target(source)
            (target/'result.json').unlink()
            (target/'checkpoint-2500').mkdir()
            with self.assertRaises(ValueError):cleanup_target(source)
            (target/'checkpoint-2500').rmdir()
            (target/'escape').symlink_to(sibling,target_is_directory=True)
            with self.assertRaises(ValueError):cleanup_target(source)
            (target/'escape').unlink()
            (target/'latest-run').symlink_to('checkpoint-250',target_is_directory=True)
            self.assertEqual(cleanup_target(source),target)
            with patch('scripts.gpu_status.require_free',side_effect=RuntimeError('busy')):
                with self.assertRaises(RuntimeError):cleanup(source,run)
            self.assertTrue(target.exists())
            external=root/'external-wandb.log';external.write_text('preserve')
            (target/'debug-core.log').symlink_to(external)
            with patch('scripts.gpu_status.require_free') as free, patch('scripts.p7_tuning_launch.verify_reference'), patch('scripts.p7_tuning_launch.EXTERNAL_LOG_LINKS',{'debug-core.log':str(external)}):
                cleanup(source,run)
            self.assertEqual(external.read_text(),'preserve')
            free.assert_called_once_with(list(range(8)))
            self.assertFalse(target.exists());self.assertTrue((sibling/'keep').exists())
            receipt=json.loads((run/'interrupted-output-before-cleanup.json').read_text())
            self.assertIn('train_config.json',receipt['json_files'])
            self.assertTrue((source/'supervised/supervisor.json').exists())

    def test_reference_gate_rejects_changed_training_order(self):
        import copy
        from scripts.p7_tuning_launch import verify_reference
        reference=dict(training=dict(seed=42,max_steps=2500,warmup_steps=125,logging_steps=10),
                       data=dict(eval_rows=4882),pilot=dict(proxy_lambda_max=0),
                       train_fingerprint='train42',eval_fingerprint='full')
        smoke=copy.deepcopy(reference)
        smoke['training']['logging_steps']=1;smoke['data']['eval_rows']=16
        smoke['pilot']['proxy_lambda_max']=.3;smoke['eval_fingerprint']='prefix'
        def read_config(path):return copy.deepcopy(reference if 'smoke' not in str(path) else smoke)
        with patch('scripts.p7_tuning_launch.read',side_effect=read_config), patch('deep_kv.report.comparable_config',side_effect=copy.deepcopy):
            verify_reference('/old','/new')
            smoke['train_fingerprint']='wrong-order'
            with self.assertRaises(ValueError):verify_reference('/old','/new')

    def test_combined_queue_gates_all_22_before_production(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            jobs=make('proxy_heads.12layer.b200.json',root/'jobs.json','rows',[c[0] for c in ALL_CASES])
            self.assertEqual(len(CASES),20);self.assertEqual(len(ALL_CASES),22)
            self.assertEqual(len(jobs),91)
            names=[j['name'] for j in jobs]
            gate=names.index('smoke-recipes')
            self.assertEqual(names[gate+1:gate+5],['lm-noaux-train','lm-noaux-validate','lm-nexttoken-train','lm-nexttoken-validate'])
            for label,arm,weight,seed in ALL_CASES:
                self.assertLess(names.index('smoke-'+label+'-validate'),gate)
                job=jobs[names.index(label+'-train')]
                self.assertEqual(job['gpus'],list(range(8)))
                self.assertNotIn('--resume_from_checkpoint',job['argv'])
            cases=json.loads((root/'jobs.config/numerical-cases.json').read_text())
            self.assertEqual(len(cases),21)
