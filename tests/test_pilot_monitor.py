"""Mock-only dev monitor checks; no Git pushes, Dropbox calls or GPU actions."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import monitor_pilot_local as monitor


class MonitorTests(unittest.TestCase):
    def test_dirty_checkout_is_not_modified(self):
        with patch.object(monitor, 'git', return_value=' M commands.sh'), \
             patch.object(monitor.subprocess, 'run') as change:
            with self.assertRaisesRegex(RuntimeError, 'checkout changed'):
                monitor.submit_snapshot(Path('/unused'), 'a'*40, 'test', '/patch')
            change.assert_not_called()

    def test_head_change_is_not_modified(self):
        with patch.object(monitor, 'git', side_effect=['', 'git@github-share:deep-llms/th2.git', '', 'b'*40]), \
             patch.object(monitor.subprocess, 'run') as change:
            with self.assertRaisesRegex(RuntimeError, 'HEAD changed'):
                monitor.submit_snapshot(Path('/unused'), 'a'*40, 'test', '/patch')
            change.assert_not_called()

    def test_only_readonly_snapshot_is_submitted(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'commands.sh').write_text('#1\n#previous\nbash old.sh\n')
            def fake_git(path, *args):
                if args[:2] == ('remote', 'get-url'):
                    return 'git@github-share:deep-llms/th2.git'
                if args[0] in ('rev-parse', 'ls-remote'):
                    return 'a'*40
                return ''
            with patch.object(monitor, 'git', side_effect=fake_git) as git, \
                 patch.object(monitor.subprocess, 'run') as apply:
                monitor.submit_snapshot(root, 'a'*40, 'test', '/patch')
                patch_text = apply.call_args.kwargs['input']
                added = [x[1:] for x in patch_text.splitlines() if x.startswith('+')]
                self.assertEqual(len(added), 2)
                self.assertTrue(added[0].startswith('#2 +a -f-'))
                self.assertTrue(all(not x.endswith('/') for x in monitor.EXPORTS))
                self.assertNotIn('force', str(git.call_args_list))


if __name__ == '__main__':
    unittest.main()
