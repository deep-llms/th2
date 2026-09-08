import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from accelerate.commands.config.config_args import load_config_from_file
from scripts.reclaim_verified_burn import reclaim, verified_workers
from scripts.verify_capacity_run import verify


class ReclaimTests(unittest.TestCase):
    def setUp(self):
        self.path = '/tmp/llm_pretrain_burn.py'
        self.rows = [dict(index=0, pids=[101]), dict(index=1, pids=[102])]
        self.procs = {pid: dict(pid=pid, parent=100, start=str(pid), argv=['python', '-c', 'worker'])
                      for pid in (101, 102)}
        self.procs[100] = dict(pid=100, parent=1, start='100', argv=['/env/bin/python', self.path])

    def test_only_compute_workers_are_signaled(self):
        with patch('scripts.reclaim_verified_burn.snapshot', return_value=self.rows), \
             patch('scripts.reclaim_verified_burn.identity', side_effect=lambda pid:self.procs[pid]), \
             patch('scripts.reclaim_verified_burn.os.kill') as kill:
            reclaim(self.path, [0,1], stop=True)
        self.assertEqual([call.args[0] for call in kill.call_args_list], [101,102])
        self.assertTrue(all(call.args[1] == 9 for call in kill.call_args_list))

    def test_readonly_and_free_are_noops(self):
        with patch('scripts.reclaim_verified_burn.snapshot', return_value=self.rows), \
             patch('scripts.reclaim_verified_burn.identity', side_effect=lambda pid:self.procs[pid]), \
             patch('scripts.reclaim_verified_burn.os.kill') as kill:
            reclaim(self.path, [0,1])
            kill.assert_not_called()
        with patch('scripts.reclaim_verified_burn.snapshot', return_value=[dict(index=0,pids=[])]), \
             patch('scripts.reclaim_verified_burn.os.kill') as kill:
            reclaim(self.path, [0], stop=True)
            kill.assert_not_called()

    def test_unknown_or_embedded_path_is_rejected_without_signals(self):
        self.procs[100]['argv'] = ['bash','-c', f'python {self.path}']
        with patch('scripts.reclaim_verified_burn.snapshot', return_value=self.rows), \
             patch('scripts.reclaim_verified_burn.identity', side_effect=lambda pid:self.procs[pid]), \
             patch('scripts.reclaim_verified_burn.os.kill') as kill:
            with self.assertRaisesRegex(RuntimeError, 'one verified burn'):
                reclaim(self.path, [0,1], stop=True)
            kill.assert_not_called()

    def test_changed_start_time_is_rejected_without_signals(self):
        workers = [self.procs[101], self.procs[102]]
        changed = [dict(workers[0], start='new'), workers[1]]
        with patch('scripts.reclaim_verified_burn.verified_workers', side_effect=[
                    (workers,self.procs[100]), (changed,self.procs[100])]), \
             patch('scripts.reclaim_verified_burn.os.kill') as kill:
            with self.assertRaisesRegex(RuntimeError, 'identities changed'):
                reclaim(self.path, [0,1], stop=True)
            kill.assert_not_called()

    def test_shared_unselected_gpu_rejected(self):
        with patch('scripts.reclaim_verified_burn.snapshot', side_effect=[self.rows,
                    self.rows + [dict(index=2, pids=[101])]]), \
             patch('scripts.reclaim_verified_burn.identity', side_effect=lambda pid:self.procs[pid]):
            with self.assertRaisesRegex(RuntimeError, 'unselected GPU'):
                verified_workers(self.path, [0,1])


class CompletionTests(unittest.TestCase):
    def test_completion_and_nonfinite_weights(self):
        self.check_completion(10000)

    def test_5000_step_completion_and_reject_wrong_cutoff(self):
        self.check_completion(5000)

    def check_completion(self, step):
        # Tiny synthetic artifact contract, not a production architecture test.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def record(path, value):
                path.write_text(json.dumps(value))
            record(root/'result.json', dict(success=True, arm='A128', global_step=step,
                schedule_steps=20000, status='stopped_at_step', parameters={'total':2},
                train_metrics={'train_loss':3.}, eval_metrics={'eval_loss':3.,'eval_scored_targets':10}))
            record(root/'train_config.json', dict(model={'tiny_test':False, 'num_hidden_layers':6},
                execution={'world_size':2, 'effective_batch_size':512}, train_fingerprint='train', eval_fingerprint='eval'))
            record(root/'cache.json', dict(train={'fingerprint':'train'}, eval={'fingerprint':'eval','scored_targets':10}))
            for name in ('final', f'checkpoint-{step}'):
                path = root/name
                path.mkdir()
                record(path/'config.json', dict(experiment_arm='A128', num_hidden_layers=6, tie_word_embeddings=False))
                for file in ('tokenizer.json','tokenizer_config.json','optimizer.pt','scheduler.pt','rng_state_0.pth','rng_state_1.pth'):
                    (path/file).write_bytes(b'fixture')
                record(path/'trainer_state.json', {'global_step':step})
                save_file({'weight':torch.ones(2)}, path/'model.safetensors')
            with patch('scripts.verify_capacity_run.EXPECTED_COUNTS', {'A128':2}):
                self.assertTrue(verify(root,'A128',step,2,root/'cache.json')['success'])
                with self.assertRaisesRegex(ValueError, 'Training result/arm/step mismatch'):
                    verify(root,'A128',step+1,2,root/'cache.json')
                save_file({'weight':torch.tensor([1.,float('nan')])}, root/'final'/'model.safetensors')
                with self.assertRaisesRegex(ValueError,'Non-finite model tensor'):
                    verify(root,'A128',step,2,root/'cache.json')

    def test_shell_syntax_and_eight_gpu_config(self):
        root = Path(__file__).resolve().parents[1]
        subprocess.run(['bash','-n',str(root/'scripts/train_capacity_b200.sh')], check=True)
        config = load_config_from_file(str(root/'resources/accelerate_config.yaml'))
        self.assertEqual(config.num_processes,8)
        self.assertEqual(config.mixed_precision,'bf16')
        self.assertEqual(config.distributed_type.value,'MULTI_GPU')
        self.assertFalse(config.use_cpu)
        script = (root/'scripts/train_capacity_b200.sh').read_text()
        self.assertIn('TASK_STOP_AT_STEP=${SWT_STOP_AT_STEP:-10000}', script)
        self.assertIn('--num_train_epochs 1 --stop-at-step "$TASK_STOP_AT_STEP"', script)
        self.assertIn('--step "$TASK_STOP_AT_STEP" --world-size 8', script)
        self.assertIn("row['step']==stop_at_step", script)
        self.assertNotIn('--max_steps', script)
        self.assertNotIn('--max-steps', script)


if __name__ == '__main__':
    unittest.main()
