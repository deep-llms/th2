import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

from scripts.benchmark_capacity_batch import MeasureUpdates, summarize, main


class BatchBenchmarkTests(unittest.TestCase):
    def test_slowest_rank_timing_and_counts(self):
        ranks = [dict(durations=[1.,2.], peak_allocated_gib=3., peak_reserved_gib=4.),
                 dict(durations=[2.,1.], peak_allocated_gib=5., peak_reserved_gib=6.)]
        result = summarize(ranks, 512, 2048)
        self.assertEqual(result['median_optimizer_step_s'], 2.)
        self.assertEqual(result['input_tokens_per_second'], 524288.)
        self.assertEqual(result['peak_allocated_gib'], 5.)
        with self.assertRaises(ValueError):
            summarize([dict(ranks[0], durations=[])], 512, 2048)

    def test_callback_excludes_warmup_and_stops_without_saving(self):
        callback = MeasureUpdates(1, 2)
        args = SimpleNamespace(device=torch.device('cpu'))
        control = SimpleNamespace(should_training_stop=False, should_save=False)
        for step in (1,2,3):
            state = SimpleNamespace(global_step=step)
            with patch('time.perf_counter', side_effect=[10.,12.]):
                callback.on_step_begin(args, state, control)
                callback.on_step_end(args, state, control)
        self.assertEqual(callback.durations, [2.,2.])
        self.assertTrue(control.should_training_stop)
        self.assertFalse(control.should_save)

    def test_real_trainer_tiny_batch_pairs(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder:
            for batch, accum in ((2,2), (4,1)):
                root = Path(folder)/f'b{batch}'
                argv = ['benchmark', '--arm', 'B0', '--batch-size', str(batch), '--accumulation', str(accum),
                        '--warmup-updates', '1', '--measure-updates', '2', '--tiny-test', '--output-dir', str(root)]
                with patch.object(sys, 'argv', argv):
                    main()
                report = json.loads((root/'result.json').read_text())
                self.assertEqual(report['effective_batch'], 4)
                self.assertEqual(len(report['metrics']['measured_step_seconds']), 2)
                self.assertFalse(list(root.glob('checkpoint-*')))


if __name__ == '__main__':
    unittest.main()
