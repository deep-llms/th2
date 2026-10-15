import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from run_experiments import run_jobs
from scripts.reorder_p7_tuning import PRIORITY, remaining_jobs, boundary_finished
from scripts.train_then_burn import digest


class ReorderTests(unittest.TestCase):
    def test_reorders_exact_jobs_and_keeps_all_fifteen(self):
        labels = ['tg-heads8', 'tg-keyonly', 'tg-valueonly', 'lm-lambda001',
                  'lm-lambda1', 'lm-sparse-noaux', 'lm-keyonly-noaux',
                  'lm-valueonly-noaux', 'lm-valueonly-gated', 'a-seed2042',
                  'lm-noaux-seed2042', 'tg-medium', 'tg-weighted', 'tg-huber', 'tg-nonlinear']
        jobs = [dict(name=label+suffix, argv=['python', label, suffix], required_outputs=[])
                for label in labels for suffix in ('-train', '-validate')]
        jobs.append(dict(name='summary', argv=['python', '--cases', 'old'], required_outputs=[]))
        original = copy.deepcopy(jobs)
        order, new = remaining_jobs(jobs)
        self.assertEqual(order[:8], list(PRIORITY))
        self.assertEqual(order[8:], [l for l in labels if l not in PRIORITY])
        self.assertEqual(jobs, original)
        self.assertEqual(len(new), 31)
        by_name = {j['name']: j for j in jobs}
        for job in new[:-1]:
            self.assertEqual(job, by_name[job['name']])
        self.assertEqual(new[-1]['argv'][2:], order)
        with self.assertRaises(ValueError):
            remaining_jobs(jobs[2:])

    def test_reserved_log_allows_current_validation_but_never_launches_next(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'
            # First job emulates a completed current arm and installs boundary.
            code = "from pathlib import Path; Path('run/current.json').write_text('{}'); Path('run/tg-heads8-train.log').write_text('authorized boundary')"
            jobs = [dict(name='tg-heads2-train', argv=[sys.executable, '-c', code],
                         required_outputs=[dict(path='current.json')]),
                    dict(name='tg-heads2-validate', argv=[sys.executable, '-c',
                         "from pathlib import Path; Path('run/validated.json').write_text('{}')"],
                         required_outputs=[dict(path='validated.json')]),
                    dict(name='tg-heads8-train', argv=[sys.executable, '-c',
                         "from pathlib import Path; Path('run/should-not-exist').write_text('bad')"],
                         required_outputs=[dict(path='should-not-exist')])]
            with self.assertRaises(FileExistsError):
                run_jobs(jobs, Path(tmp), root)
            state = json.loads((root/'run.json').read_text())
            self.assertEqual([j['status'] for j in state['jobs']], ['ok', 'ok', 'failed'])
            self.assertNotIn('returncode', state['jobs'][-1])
            self.assertFalse((root/'should-not-exist').exists())
            self.assertFalse((root/'complete.json').exists())

    def test_handoff_does_not_accept_training_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            (source/'supervised/run').mkdir(parents=True)
            (source/'supervised/run/run.json').write_text(json.dumps(dict(status='failed',
                error='training failed', jobs=[dict(name='tg-heads2-train', status='failed')])) )
            (source/'supervised/supervisor.json').write_text(json.dumps(dict(finished_at='now')))
            with self.assertRaises(ValueError):
                boundary_finished(source, source)

    def test_requires_verified_burn_after_expected_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'old'
            root = Path(tmp)/'new'
            root.mkdir()
            run = source/'supervised/run'
            run.mkdir(parents=True)
            barrier = run/'tg-heads8-train.log'
            barrier.write_text('authorized boundary')
            (root/'boundary-installed.json').write_text(json.dumps(dict(path=str(barrier), sha256=digest(barrier))))
            state = dict(status='failed', error='[Errno 17] File exists: next log', jobs=[
                dict(name='tg-heads2-train', status='ok'),
                dict(name='tg-heads2-validate', status='ok'),
                dict(name='tg-heads8-train', status='failed')])
            (run/'run.json').write_text(json.dumps(state))
            burn = dict(collective_progress_verified=True, session='expected')
            receipt = dict(training_status='failed', finished_at='now', burn=burn)
            (source/'supervised/supervisor.json').write_text(json.dumps(receipt))
            (source/'supervised/burn-verified.json').write_text(json.dumps(burn))
            original = Path.exists
            def exists(path):
                return False if str(path)=='/mnt/local/_gpu_guard/DISABLED' else original(path)
            with patch.object(Path, 'exists', exists):
                self.assertTrue(boundary_finished(source, root))
                receipt['burn']['collective_progress_verified'] = False
                (source/'supervised/supervisor.json').write_text(json.dumps(receipt))
                with self.assertRaises(ValueError):
                    boundary_finished(source, root)


if __name__ == '__main__':
    unittest.main()
