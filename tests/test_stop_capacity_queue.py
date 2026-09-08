import unittest
from unittest.mock import patch
from scripts.stop_capacity_queue import identify, unchanged


class StopQueueTests(unittest.TestCase):
    def records(self):
        root = '/mnt/local/_outputs/deep-llms_th2/swt/fixture'
        queue = dict(pid=10, start='100', parent=3,
            argv=['bash', 'scripts/train_capacity_b200.sh', root, 'B0','A128','A256','A512','C','D'])
        launcher = dict(pid=20, start='200', parent=10,
            argv=['python', '/mnt/local/conda-py311/envs/swt/bin/accelerate', 'launch'])
        workers = [dict(pid=i+30, parent=20, start=str(i+300), argv=[
            '/mnt/local/conda-py311/envs/swt/bin/python3.11', '-u', 'train.py', '--output_dir', root+'/B0'])
            for i in range(8)]
        return root, {p['pid']:p for p in [queue, launcher, *workers]}

    def test_identity_and_output_are_required(self):
        root, records = self.records()
        status = [dict(index=i, pids=[i+30]) for i in range(8)]
        with patch('scripts.stop_capacity_queue.identity', side_effect=lambda pid: records[pid]), \
             patch('scripts.stop_capacity_queue.snapshot', return_value=status), \
             patch('os.kill') as signal:
            queue, launcher, workers = identify(10, '100', root)
            self.assertEqual(len(workers), 8)
            signal.assert_not_called()
            with self.assertRaises(RuntimeError):
                identify(10, 'WRONG', root)
            records[30]['argv'][-1] = '/other/run/B0'
            with self.assertRaises(RuntimeError):
                identify(10, '100', root)

    def test_reparenting_allowed_but_not_pid_reuse(self):
        original = dict(pid=10, start='100', parent=2, argv=['worker'])
        with patch('scripts.stop_capacity_queue.identity', return_value=dict(original, parent=1)):
            self.assertTrue(unchanged(original))
        with patch('scripts.stop_capacity_queue.identity', return_value=dict(original, start='101')):
            self.assertFalse(unchanged(original))
