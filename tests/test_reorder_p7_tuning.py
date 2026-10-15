import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from run_experiments import run_jobs
from scripts.reorder_p7_tuning import PRIORITY, remaining_jobs, boundary_finished


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


if __name__ == '__main__':
    unittest.main()
