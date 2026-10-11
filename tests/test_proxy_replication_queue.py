import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.proxy_replication_queue import make, predecessor_ready, prepare


class ReplicationTests(unittest.TestCase):
    def test_seed42_pair_preserves_schedule_and_appends_cooldown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            recipe=json.loads(Path('proxy_heads.b200.json').read_text())
            recipe['proxy_sync_optimization']=True
            path=root/'recipe.json';path.write_text(json.dumps(recipe))
            items=make(path,root/'jobs.json',42)
            self.assertEqual([j['name'] for j in items],[
                'seed-42-arm-A','seed-42-validate-A','seed-42-arm-P6-iso','seed-42-validate-P6-iso',
                'seed-42-compare','prepare-cooldowns','cooldown-A','cooldown-P6-iso','validate-cooldowns'])
            for job in [j for j in items if 'gpus' in j]:
                self.assertEqual(job['gpus'],list(range(8)))
                self.assertEqual(job['argv'][:2],['env','PYTHONHASHSEED=42'])
            for job in (items[0],items[2]):
                args=job['argv']
                for flag,value in dict(seed='42',data_seed='42',stop_after='10000',max_steps='28600',
                    warmup_steps='1430',save_total_limit='0',proxy_sync_optimization='true').items():
                    self.assertEqual(args[args.index('--'+flag)+1],value)
                self.assertNotIn('--resume_from_checkpoint',args)
            self.assertNotIn('--proxy_module_seed',items[0]['argv'])
            args=items[2]['argv'];self.assertEqual(args[args.index('--proxy_module_seed')+1],'43')

    def test_handoff_rejects_failed_or_unverified_predecessor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);s=root/'supervised';(s/'run/branches').mkdir(parents=True)
            guard=root/'guard';state=s/'supervisor.json'
            self.assertFalse(predecessor_ready(root,guard))
            state.write_text(json.dumps(dict(training_status='running')))
            self.assertFalse(predecessor_ready(root,guard))
            state.write_text(json.dumps(dict(training_status='failed')))
            with self.assertRaises(RuntimeError):predecessor_ready(root,guard)
            success=dict(training_status='ok',training_returncode=0,finished_at='done',
                         burn=dict(collective_progress_verified=True,session='verified-burn'))
            state.write_text(json.dumps(success))
            (s/'run/run.json').write_text(json.dumps(dict(status='ok')))
            summary=s/'run/branches/summary.json'
            summary.write_text(json.dumps(dict(status='passed',original_checkpoints_unchanged=True)))
            burn=s/'burn-verified.json'
            burn.write_text(json.dumps(dict(collective_progress_verified=True,session='wrong-burn')))
            with self.assertRaises(RuntimeError):predecessor_ready(root,guard)
            burn.write_text(json.dumps(success['burn']))
            guard.touch();self.assertFalse(predecessor_ready(root,guard))
            guard.unlink();self.assertTrue(predecessor_ready(root,guard))
            summary.write_text(json.dumps(dict(status='passed',original_checkpoints_unchanged=False)))
            with self.assertRaises(RuntimeError):predecessor_ready(root,guard)

    def test_prepare_preserves_original_and_rejects_wrong_module_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for arm in ('A','P6-iso'):
                source=root/'seed-42'/arm;source.mkdir(parents=True)
                (source/'train_config.json').write_text(json.dumps(dict(
                    training=dict(seed=42,data_seed=42,save_total_limit=0),
                    pilot=dict(proxy_sync_optimization=True,proxy_module_seed=43))))
                (source/'result.json').write_text(json.dumps(dict(global_step=10000,arm=arm)))
            def staged(folder,arm,checkpoint,steps,optimized):
                self.assertEqual(checkpoint,root/'seed-42'/arm/'checkpoint-10000')
                self.assertEqual((steps,optimized),(1000,True));folder.mkdir(parents=True,exist_ok=True)
            with patch('scripts.proxy_replication_queue.branch',side_effect=staged) as stage:
                prepare(root,42);self.assertEqual(stage.call_count,2)
                cfg=root/'seed-42/P6-iso/train_config.json';data=json.loads(cfg.read_text())
                data['pilot']['proxy_module_seed']=1043;cfg.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError,'proxy initialization'):prepare(root,42)


if __name__=='__main__':unittest.main()
