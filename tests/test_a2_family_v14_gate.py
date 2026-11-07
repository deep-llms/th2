import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import a2_family_v14_gate as gate
from scripts.a2_family_v14_install_gate import bridge_bytes


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def completed(self):
        artifact = self.root / 'report.txt'
        artifact.write_text('scientific outputs, including all flags')
        lock = dict(path=str(artifact), sha256=gate.sha(artifact))
        names = ['ST1-dclm-S72-S1-0.2B-lr1e-3'] + [
            f'ST1-{ds}-{arm}-S1-0.8B-lr1e-3'
            for ds in ('dclm', 'fwe') for arm in ('S72', 'S188')]
        return dict(spec_sha256=gate.SPEC_SHA, execution_complete=True,
            report_complete=True, backup_verified=True, extension_dataset=None,
            events=[dict(name=n, returncode=0) for n in names],
            report=lock, decision=lock, backup_files=[lock])

    def check_value(self, value):
        p = self.root / 'f6-complete.json'
        p.write_text(json.dumps(value))
        return gate.verify_completion(p)

    def test_no_readiness_cannot_reach_depth_or_launch(self):
        with patch.object(gate.time, 'sleep', side_effect=InterruptedError), patch.object(gate.subprocess, 'run') as launch:
            with self.assertRaises(InterruptedError):
                gate.run_gate(self.root, 0)
        launch.assert_not_called()
        self.assertFalse(json.loads((self.root / 'gate-status.json').read_text())['depth_resume_allowed'])

    def test_missing_row_and_duplicate_row_rejected(self):
        v = self.completed(); v['events'].pop()
        with self.assertRaises(ValueError): self.check_value(v)
        v = self.completed(); v['events'].append(v['events'][0])
        with self.assertRaises(ValueError): self.check_value(v)

    def test_extension_requires_both_and_only_selected_dataset(self):
        v = self.completed(); v['extension_dataset'] = 'fwe'
        with self.assertRaises(ValueError): self.check_value(v)
        for arm in ('S72', 'S188'):
            v['events'].append(dict(name=f'ST1-fwe-{arm}-S1-3.2B-lr1e-3', returncode=0))
        self.check_value(v)
        v['events'].append(dict(name='ST1-dclm-S72-S1-3.2B-lr1e-3', returncode=0))
        with self.assertRaises(ValueError): self.check_value(v)

    def test_corrupt_backup_or_missing_report_never_clears(self):
        v = self.completed(); Path(v['backup_files'][0]['path']).write_text('corrupt')
        with self.assertRaises(ValueError): self.check_value(v)
        v = self.completed(); v['report_complete'] = False
        with self.assertRaises(ValueError): self.check_value(v)

    def test_scientific_failure_recorded_not_hidden(self):
        v = self.completed(); v['events'][1]['returncode'] = 1
        self.assertEqual(self.check_value(v)['events'][1]['returncode'], 1)

    def test_failed_attempt_cannot_implicitly_relaunch(self):
        ready = self.root / 'ready.json'; ready.write_text('{}')
        (self.root / 'attempt.json').write_text(json.dumps(dict(ready_sha256=gate.sha(ready))))
        with patch.object(gate, 'verify_ready', return_value=(self.root, {})), patch.object(gate.subprocess, 'run') as launch:
            with self.assertRaisesRegex(ValueError, 'no blind relaunch'):
                gate.run_gate(self.root)
        launch.assert_not_called()

    def test_coordinator_failure_blocks_depth(self):
        (self.root / 'ready.json').write_text(json.dumps(dict(config=dict(path='config.json'))))
        with patch.object(gate, 'verify_ready', return_value=(self.root, {})), patch.object(gate.subprocess, 'run') as launch:
            launch.return_value.returncode = 1
            with self.assertRaises(RuntimeError): gate.run_gate(self.root)
        self.assertFalse(json.loads((self.root / 'gate-status.json').read_text())['depth_resume_allowed'])

    def test_bridge_hook_precedes_resume_and_preserves_original_code(self):
        original = Path('scripts/a2_family_depth_bridge.py').read_bytes()
        result = bridge_bytes(original, self.root / 'gate.py', 'a' * 64, self.root)
        compile(result, 'bridge', 'exec')
        anchor = b'    record,plan,resume=validate(cfg);require_free(list(range(8)))'
        before, after = original.split(anchor)
        self.assertTrue(result.startswith(before))
        self.assertTrue(result.endswith(anchor + after))
        self.assertLess(result.index(b"['run_gate']"), result.index(anchor))

    def test_unrecognized_bridge_is_not_patched(self):
        with self.assertRaises(ValueError):
            bridge_bytes(b'# changed bridge', self.root / 'gate.py', 'a' * 64, self.root)


if __name__ == '__main__':
    unittest.main()
