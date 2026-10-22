"""Seed-only configuration and comparison contracts; no GPUs or model imports."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.bdot_router_confirm import ARMS, check, prepare, read, remap, report
from run_experiments import load_jobs

ROOT = Path(__file__).resolve().parents[1]


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prior, self.out = self.root / 'prior', self.root / 'new'
        self.out.mkdir()
        subprocess.run([sys.executable, str(ROOT / 'scripts/make_bdot_pilot.py'),
            '--study', 'router', '--output', str(self.prior / 'prepared'),
            '--tokenizer', str(self.root / 'tokenizer'), '--data', str(self.root / 'train'),
            '--eval-data', str(self.root / 'val'), '--revision', 'a'*40,
            '--accelerate-config', str(self.prior / 'source/resources/accelerate_b200.yaml'),
            '--expert-backend', 'grouped'], check=True, capture_output=True)
        jobs = read(self.prior / 'prepared/jobs.json')
        jobs = remap(jobs, [(str(ROOT), str(self.prior / 'source'))])
        put(self.prior / 'prepared/jobs.json', jobs)
        original = read(self.prior / 'prepared' / (ARMS[0] + '-s0.train.json'))
        self.base = self.root / 'D-s1.train.json'
        self.base1 = self.root / 'dense1'
        put(self.base, dict(original, seed=1, output_dir=str(self.base1), config_name='/dense.json'))
        self.driver = self.root / 'confirmation.py'

    def prepared(self):
        return prepare(self.out, self.prior, self.base, self.driver)

    def test_exact_seed_only_queue(self):
        jobs = read(self.prepared())['jobs']
        load_jobs(self.out / 'gated-jobs.json')
        self.assertEqual([j['name'] for j in jobs], ['D-s1-carry', 'confirmation_check',
            ARMS[0]+'-s1', ARMS[0]+'-s1-eval', ARMS[1]+'-s1', ARMS[1]+'-s1-eval', 'confirmation_report'])
        for arm in ARMS:
            old = read(self.prior / 'prepared' / (arm + '-s0.train.json'))
            new = read(self.out / 'prepared' / (arm + '-s1.train.json'))
            self.assertEqual({k for k in old if old[k] != new[k]}, {'seed', 'config_name', 'output_dir'})
            self.assertEqual((new['seed'], new['data_seed']), (1, 0))
            old_model, new_model = read(old['config_name']), read(new['config_name'])
            self.assertEqual(new_model['bdot'].pop('seed'), 1)
            old_model['bdot'].pop('seed')
            self.assertEqual(old_model, new_model)
            training = next(j for j in jobs if j['name'] == arm + '-s1')
            self.assertEqual(training['argv'][-1], str(self.out / 'prepared' / (arm + '-s1.train.json')))
            self.assertIn(str(self.prior / 'source/train.py'), training['argv'])
            evaluation = next(j for j in jobs if j['name'] == arm + '-s1-eval')
            self.assertEqual(evaluation['argv'][evaluation['argv'].index('--routing_seed')+1], '1')
            for job in (training, evaluation):
                self.assertEqual(job['gpus'], list(range(8)))
                self.assertFalse(any(arm + '-s0' in str(v) for v in job['argv']))
        with self.assertRaises(FileExistsError):
            self.prepared()

    def test_rejects_changed_data_seed(self):
        base = read(self.base); base['data_seed'] = 1; put(self.base, base)
        with self.assertRaisesRegex(ValueError, 'seed-1 dense recipe'):
            self.prepared()

    def test_rejects_unmatched_recipe(self):
        p = self.prior / 'prepared' / (ARMS[1] + '-s0.train.json')
        parent = read(p); parent['learning_rate'] *= 2; put(p, parent)
        with self.assertRaisesRegex(ValueError, 'Unrelated recipe changes'):
            self.prepared()

    def result_fixture(self):
        self.prepared()
        base0 = self.root / 'pilot/prepared/runs/D'
        put(self.prior / 'followup.json', dict(pilot=str(self.root / 'pilot')))
        for base, loss in [(base0, 3.49), (self.base1, 3.50)]:
            put(base / 'final_eval.json', dict(eval_loss=loss))
            put(base / 'data_recipe.json', dict(fingerprint='same-order'))
        for arm in ARMS:
            for seed, run in [(0, self.prior), (1, self.out)]:
                name = arm + '-s' + str(seed); loss = 3.49 + .01*seed + .002
                put(run / 'prepared/runs' / name / 'final_eval.json', dict(eval_loss=loss))
                put(run / 'prepared/runs' / name / 'data_recipe.json', dict(fingerprint='same-order'))
                put(run / 'supervised/run' / name / 'result.json', dict(status='ok', global_step=4768,
                    loss=loss, num_tokens=9915698, num_sequences=9704))
                put(run / 'supervised/run' / (name+'.verified.json'), dict(status='ok', global_step=4768))

    def test_reports_paired_seed_baselines(self):
        self.result_fixture()
        report(self.out)
        result = read(self.out / 'supervised/run/confirmation-results.json')
        self.assertEqual(result['status'], 'ok')
        for arm in result['arms']:
            self.assertAlmostEqual(arm['mean_gap_to_D'], .002)
            self.assertEqual(len(arm['seed_losses']), 2)
            self.assertTrue(arm['within_001_nats_both_seeds'])

    def test_rejects_changed_final_data(self):
        self.result_fixture()
        put(self.out / 'prepared/runs' / (ARMS[0]+'-s1') / 'data_recipe.json', dict(fingerprint='different'))
        with self.assertRaisesRegex(ValueError, 'Invalid comparison'):
            report(self.out)

    def test_carry_check_fails_before_training(self):
        self.prepared()
        put(self.base1 / 'final_eval.json', dict(eval_loss=3.5))
        result = dict(status='ok', global_step=4768, loss=3.51, num_tokens=9915698, num_sequences=9704)
        path = self.out / 'supervised/run/D-s1-carry/result.json'
        put(path, result)
        with self.assertRaisesRegex(ValueError, 'does not reproduce'):
            check(self.out)
        result['loss'] = 3.5; put(path, result); check(self.out)
        self.assertEqual(read(self.out / 'supervised/run/confirmation-check.json')['status'], 'ok')


if __name__ == '__main__':
    unittest.main()
