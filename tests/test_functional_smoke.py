"""CPU-only checks that the production smoke gate rejects incomplete evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from scripts.smoke_deep_kv_functional import validate_arm
from deep_kv import BOTTLENECK_ARMS, code_loss_weight


class FunctionalSmokeGateTests(unittest.TestCase):
    def fixture(self, root, arm='G', step=10):
        def write(name, value):
            (root / name).write_text(json.dumps(value))
        metrics = dict(eval_rows=129, eval_route_queries=129 * 2047, eval_lm_loss=4.,
                       eval_loss_route=.8, eval_loss_msg=.2 if arm == 'G' else 0.)
        metrics['eval_loss'] = 4. + .3 * (metrics['eval_loss_route'] + metrics['eval_loss_msg'])
        write('result.json', dict(arm=arm, global_step=step, schedule_steps=12,
                                  status='stopped' if step == 10 else 'complete',
                                  input_tokens=step * 1048576, evaluation=metrics))
        write('trainer_state.json', dict(global_step=step, max_steps=12))
        write('train_config.json', dict(world_size=8, tokens_per_update=1048576,
              model_config=dict(num_hidden_layers=28), data=dict(block_size=2048),
              training=dict(bf16=True, per_device_train_batch_size=16, gradient_accumulation_steps=4)))
        checkpoint = root / f'checkpoint-{step}'
        checkpoint.mkdir()
        for name in ['optimizer.pt', 'scheduler.pt', *[f'rng_state_{r}.pth' for r in range(8)]]:
            (checkpoint / name).write_bytes(b'fixture')
        for rank in range(8):
            write(f'smoke-step{step}-rank{rank}.json', dict(arm=arm, rank=rank, step=step,
                  steps=[dict(step=i, seconds=3.5, loss_route=.8, loss_msg=metrics['eval_loss_msg'])
                         for i in range(1 if step == 10 else 11, step + 1)],
                  allocated_peak_bytes=30 * 2**30, reserved_peak_bytes=40 * 2**30,
                  device_total_bytes=180 * 2**30,
                  gpu_sample=[dict(index=i, pids=[100+i], memory_total_mib=180000,
                                   memory_used_mib=45000) for i in range(8)] if rank == 0 else None))

    def test_initial_and_resumed_contracts(self):
        for arm in 'FG':
            for step in (10, 12):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    self.fixture(root, arm, step)
                    record = validate_arm(root, arm, step)
                    self.assertEqual(record['seconds_per_update'], 3.5 if step == 10 else None)

    def test_invalid_loss_normalization_or_missing_rank_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            path = root / 'result.json'
            result = json.loads(path.read_text())
            result['evaluation']['eval_loss'] = 4. + .3 * (.8 + .2) / 2
            path.write_text(json.dumps(result))
            with self.assertRaises(AssertionError):
                validate_arm(root, 'G', 10)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            (root / 'smoke-step10-rank7.json').unlink()
            with self.assertRaises(FileNotFoundError):
                validate_arm(root, 'G', 10)

    def test_fake_resume_and_inadequate_memory_rejected(self):
        for field, value in [('steps', [dict(step=12, seconds=1)]),
                             ('reserved_peak_bytes', 175 * 2**30)]:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.fixture(root, step=12)
                path = root / 'smoke-step12-rank3.json'
                record = json.loads(path.read_text()); record[field] = value
                path.write_text(json.dumps(record))
                with self.assertRaises(AssertionError):
                    validate_arm(root, 'G', 12)

    def test_bottleneck_gate_and_incorrect_query_count(self):
        for arm in BOTTLENECK_ARMS:
            for step in (10, 12):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    self.fixture(root, arm, step)
                    path = root / 'result.json'
                    result = json.loads(path.read_text())
                    consumer = arm.startswith('Consumer')
                    metrics = dict(eval_rows=129, eval_lm_loss=4., eval_loss=4., eval_loss_align=.5,
                                   eval_extractor_targets=129 * (2046 if consumer else 2047),
                                   eval_objective=6. + code_loss_weight(arm) * .5)
                    metrics['eval_loss_use' if consumer else 'eval_loss_extract'] = 2.
                    if consumer:
                        metrics.update(eval_loss_use_no_message=2.2, eval_message_ce_gain=.2)
                    result.update(evaluation=metrics, training_cost={'parameters': dict(total=120, inference=100, training_only=20)})
                    path.write_text(json.dumps(result))
                    for rank in range(8):
                        record_path = root / f'smoke-step{step}-rank{rank}.json'
                        record = json.loads(record_path.read_text())
                        for row in record['steps']:
                            row.update(lm_loss=4., objective=metrics['eval_objective'], loss_align=.5,
                                       **{'loss_use' if consumer else 'loss_extract': 2.})
                        record_path.write_text(json.dumps(record))
                    validate_arm(root, arm, step)
                    result['evaluation']['eval_extractor_targets'] += 129
                    path.write_text(json.dumps(result))
                    with self.assertRaises(AssertionError):
                        validate_arm(root, arm, step)


if __name__ == '__main__':
    unittest.main()
