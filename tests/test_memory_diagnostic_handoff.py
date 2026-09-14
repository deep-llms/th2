"""Mock-only dispatcher lifecycle tests: no live GPU/process management."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


class HandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
        cls.m = importlib.import_module('pilot_memory_diagnostics')

    def test_launch_failure_waits_for_already_launched_child(self):
        m = self.m
        child = Mock()
        child.wait.return_value = 0
        with tempfile.TemporaryDirectory() as tmp, patch.object(m, 'OUT', Path(tmp)), \
                patch.object(m, 'require_free'), patch.object(m.subprocess, 'Popen',
                side_effect=[child, OSError('synthetic launch failure')]):
            with self.assertRaises(OSError):
                m.panel(True)
            child.wait.assert_called_once_with()

    def test_failure_waits_all_three_without_signals(self):
        m = self.m
        children = [Mock(), Mock(), Mock()]
        for child, code in zip(children, (1, 0, 0)):
            child.wait.return_value = code
        with tempfile.TemporaryDirectory() as tmp, patch.object(m, 'OUT', Path(tmp)), \
                patch.object(m, 'require_free'), patch.object(m.subprocess, 'Popen', side_effect=children):
            with self.assertRaises(RuntimeError):
                m.panel(False)
            for child in children:
                child.wait.assert_called_once_with()
                child.kill.assert_not_called()
                child.terminate.assert_not_called()

    def test_happy_panel_and_physical_gpu_assignments(self):
        m = self.m
        children = [Mock(), Mock(), Mock()]
        for child in children:
            child.wait.return_value = 0
        with tempfile.TemporaryDirectory() as tmp, patch.object(m, 'OUT', Path(tmp)), \
                patch.object(m, 'require_free'), patch.object(m.subprocess, 'Popen', side_effect=children) as launch:
            for arm in m.ARMS:
                out = Path(tmp)/'smoke'/arm
                out.mkdir(parents=True)
                (out/'diagnostics.json').write_text(json.dumps(dict(smoke=True, arm=arm)))
            m.panel(True)
            self.assertEqual([call.kwargs['env']['CUDA_VISIBLE_DEVICES'] for call in launch.call_args_list],
                             ['0', '1', '2'])
            self.assertTrue(all('--max-batches' in call.args[0] for call in launch.call_args_list))


if __name__ == '__main__':
    unittest.main()
