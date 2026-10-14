import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.proxy_screen_evaluation import ARMS, PROTOCOLS, make_jobs, predecessor_ready, run
from scripts.proxy_overnight_queue import CASES
from eval.benchmarks import resolve_tasks
from run_experiments import load_jobs


class ScreenEvaluationTests(unittest.TestCase):
    def test_completion_and_burn_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); sub = root / 'supervised'; (sub / 'run').mkdir(parents=True)
            guard = root / 'disabled'
            def put(name, obj):
                (sub / name).write_text(json.dumps(obj))
            self.assertFalse(predecessor_ready(root, guard))
            put('supervisor.json', dict(training_status='running'))
            self.assertFalse(predecessor_ready(root, guard))
            burn = dict(collective_progress_verified=True, session='verified-burn')
            state = dict(training_status='ok', training_returncode=0, finished_at='now', burn=burn)
            put('supervisor.json', state); put('burn-verified.json', burn)
            jobs = [dict(name=f'{label}-{kind}', status='ok') for label, _, _ in CASES
                    for kind in ('train', 'validate')]
            put('run/run.json', dict(status='ok', jobs=jobs)); put('run/summary.json', dict(status='passed'))
            put('run/complete.json', {})
            self.assertTrue(predecessor_ready(root, guard))
            guard.touch(); self.assertFalse(predecessor_ready(root, guard)); guard.unlink()
            jobs[-1]['status'] = 'failed'; put('run/run.json', dict(status='ok', jobs=jobs))
            with self.assertRaises(RuntimeError): predecessor_ready(root, guard)
            jobs[-1]['status'] = 'ok'; put('run/run.json', dict(status='ok', jobs=jobs))
            put('burn-verified.json', {**burn, 'session': 'different'})
            with self.assertRaises(RuntimeError): predecessor_ready(root, guard)
            put('supervisor.json', dict(training_status='failed'))
            with self.assertRaises(RuntimeError): predecessor_ready(root, guard)

    def test_fixed_scope_and_job_contract(self):
        self.assertEqual(len(ARMS), 6)
        self.assertEqual(PROTOCOLS['five'], (5, ['piqa', 'arc_easy', 'winogrande', 'boolq']))
        tasks = resolve_tasks([t for _, ts in PROTOCOLS.values() for t in ts], True)
        self.assertEqual(len(tasks), 75)  # Eight other tasks + 67 BLiMP subsets.
        self.assertEqual(len(set(tasks)), 75)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'jobs.json'
            path.write_text(json.dumps(dict(jobs=make_jobs(Path(tmp) / 'scope.json'))))
            self.assertEqual(len(load_jobs(path)), 3)
        jobs = make_jobs('/scope.json')
        self.assertEqual([j['name'] for j in jobs], ['numerics', 'smoke', 'full'])
        self.assertTrue(all(j['gpus'] == list(range(8)) for j in jobs))

    def test_numerics_fail_closed_on_wrong_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest = root / 'manifest.json'; manifest.write_text('{}')
            from eval.models import file_hash
            ck = [f'/checkpoints/{arm}' for arm in ARMS]
            scope = dict(checkpoints=ck, specs={c:dict(arm=a,step=2500,checkpoint_sha256='correct')
                for c,a in zip(ck,ARMS)}, protocols=PROTOCOLS, manifest=str(manifest),
                manifest_sha256=file_hash(manifest))
            path = root / 'scope.json'; path.write_text(json.dumps(scope))
            def execute(jobs, workers):
                self.assertEqual(workers, 8)
                for job,c in zip(jobs,ck):
                    Path(job['output']).write_text(json.dumps(dict(status='passed',**scope['specs'][c])))
            with patch('scripts.proxy_screen_evaluation.suite.execute', side_effect=execute):
                run('numerics', root / 'ok', path)
            self.assertEqual(json.loads((root/'ok/summary.json').read_text())['status'],'passed')
            def bad(jobs, workers):
                execute(jobs, workers)
                Path(jobs[0]['output']).write_text(json.dumps(dict(status='passed',arm='A',step=2500,checkpoint_sha256='wrong')))
            with patch('scripts.proxy_screen_evaluation.suite.execute', side_effect=bad):
                with self.assertRaises(ValueError):run('numerics', root / 'bad', path)
            self.assertFalse((root/'bad/summary.json').exists())


if __name__ == '__main__': unittest.main()
