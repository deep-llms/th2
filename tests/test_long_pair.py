"""Offline contracts for the expensive fresh-schedule training pair."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from run_experiments import load_jobs
from scripts.bdot_long_pair import ARMS, STEPS, check, prepare, read, replace_strings, report

ROOT = Path(__file__).resolve().parents[1]


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class LongPairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.router = self.root/'router/prepared'
        self.dense = self.root/'pilot/prepared'
        self.source = self.root/'router/source'
        self.output = self.root/'new'; self.output.mkdir()
        for out, args in [(self.router, ['--study', 'router']), (self.dense, ['--arms', 'D'])]:
            subprocess.run([sys.executable, str(ROOT/'scripts/make_bdot_pilot.py'), *args,
                '--output', str(out), '--tokenizer', str(self.root/'tokenizer'),
                '--data', str(self.root/'train'), '--eval-data', str(self.root/'val'),
                '--manifest', str(self.root/'validation.json'), '--revision', 'a'*40,
                '--accelerate-config', str(self.source/'resources/accelerate_b200.yaml'),
                '--expert-backend', 'grouped'], check=True, capture_output=True)
        put(self.router/'jobs.json', replace_strings(read(self.router/'jobs.json'),
                                                     [(str(ROOT), str(self.source))]))
        self.dense_recipe = self.dense/'D.train.json'
        self.router_recipe = self.router/'Partial-25-cosine-s0.train.json'

    def prepared(self):
        return prepare(self.output, self.source, self.dense_recipe, self.router_recipe,
                       self.output/'driver.py')

    def test_fresh_long_schedule_and_unchanged_models(self):
        jobs = load_jobs(self.prepared())
        self.assertEqual([j['name'] for j in jobs], ['D-carry', 'Partial-25-cosine-carry',
            'long_pair_check', 'D-s0', 'D-s0-eval', 'Partial-25-cosine-s0',
            'Partial-25-cosine-s0-eval', 'long_pair_report'])
        for arm, recipe in zip(ARMS, [self.dense_recipe, self.router_recipe]):
            name = arm+'-s0'
            old = read(recipe)
            t = read(self.output/'prepared'/(name+'.train.json'))
            self.assertEqual({k for k in old.keys() | t.keys() if old.get(k) != t.get(k)},
                {'output_dir', 'config_name', 'max_steps', 'warmup_steps'} |
                ({'keep_checkpoint_steps'} if 'keep_checkpoint_steps' in old else set()))
            self.assertEqual(read(t['config_name']), read(old['config_name']))
            self.assertEqual((t['max_steps'], t['warmup_steps']), (10000, 200))
            self.assertEqual((t['seed'], t['data_seed']), (0, 0))
            self.assertNotIn('resume_from_checkpoint', t)
            self.assertNotIn('keep_checkpoint_steps', t)
            self.assertIsNone(t['save_total_limit'])
            self.assertEqual(t['save_steps'], 250)
            train = next(j for j in jobs if j['name'] == name)
            self.assertEqual(train['argv'][train['argv'].index('--steps')+1], '10000')
            self.assertEqual(train['argv'][-1], str(self.output/'prepared'/(name+'.train.json')))
            self.assertIn(str(self.source/'train.py'), train['argv'])
            evaluation = next(j for j in jobs if j['name'] == name+'-eval')
            self.assertEqual(evaluation['argv'][evaluation['argv'].index('--checkpoint')+1],
                str(self.output/'prepared/runs'/name/'checkpoint-10000'))
            self.assertEqual(evaluation['argv'][evaluation['argv'].index('--routing_seed')+1], '0')
            for job in [train, evaluation]:
                self.assertEqual(job['gpus'], list(range(8)))
                self.assertFalse(any('checkpoint-4768' in value for value in job['argv']))
        with self.assertRaises(FileExistsError):
            self.prepared()

    def test_parent_data_difference_rejected(self):
        value = read(self.router_recipe); value['data_seed'] = 1
        put(self.router_recipe, value)
        with self.assertRaisesRegex(ValueError, 'Unmatched parent'):
            self.prepared()

    def test_inherited_resume_is_rejected(self):
        for p in [self.dense_recipe, self.router_recipe]:
            v = read(p); v['resume_from_checkpoint'] = '/old/checkpoint-4768'; put(p, v)
        with self.assertRaisesRegex(ValueError, 'Unexpected parent'):
            self.prepared()

    def test_wrong_architecture_rejected(self):
        p = Path(read(self.router_recipe)['config_name']); v = read(p)
        v['bdot']['router_detach_input'] = True; put(p, v)
        with self.assertRaisesRegex(ValueError, 'architecture'):
            self.prepared()

    def fixture(self):
        self.prepared()
        for arm, recipe in zip(ARMS, [self.dense_recipe, self.router_recipe]):
            old = Path(read(recipe)['output_dir'])
            put(old/'data_recipe.json', {'fingerprint': 'same'})
            put(old/'final_eval.json', {'eval_loss': 3.5})
            put(self.output/'supervised/run'/(arm+'-carry/result.json'), dict(
                status='ok', global_step=4768, loss=3.5, num_tokens=9915698, num_sequences=9704))
            name = arm+'-s0'; root = self.output/'prepared/runs'/name
            loss = 3.3 if arm == 'D' else 3.29
            put(root/'data_recipe.json', {'fingerprint': 'same'})
            put(root/'final_eval.json', {'eval_loss': loss})
            put(root/'train_results.json', dict(global_step=STEPS, schedule_steps=STEPS,
                train_input_tokens=STEPS*524288, train_runtime=7200))
            put(self.output/'supervised/run'/name/'result.json', dict(status='ok',
                global_step=STEPS, loss=loss, perplexity=math.exp(loss),
                num_tokens=9915698, num_sequences=9704))
            put(self.output/'supervised/run'/(name+'.verified.json'), dict(status='ok', global_step=STEPS))
            for step in range(250, STEPS+1, 250):
                (root/f'checkpoint-{step}').mkdir()

    def test_carry_failure_blocks_training(self):
        self.fixture()
        p = self.output/'supervised/run/D-carry/result.json'; v = read(p)
        v['loss'] += .01; put(p, v)
        with self.assertRaisesRegex(ValueError, 'carry evaluation failed'):
            check(self.output)
        self.assertFalse((self.output/'supervised/run/long-pair-check.json').exists())

    def test_reports_matched_new_dense_control(self):
        self.fixture(); check(self.output); report(self.output)
        value = read(self.output/'supervised/run/long-pair-report.json')
        self.assertAlmostEqual(value['routed_gap_to_D'], -.01)
        self.assertEqual(value['steps'], 10000)

    def test_missing_intermediate_checkpoint_rejected(self):
        self.fixture()
        (self.output/'prepared/runs/D-s0/checkpoint-500').rmdir()
        with self.assertRaisesRegex(ValueError, 'checkpoint retention'):
            report(self.output)

    def test_changed_final_data_rejected(self):
        self.fixture()
        put(self.output/'prepared/runs/Partial-25-cosine-s0/data_recipe.json', {'fingerprint': 'different'})
        with self.assertRaisesRegex(ValueError, 'Invalid final'):
            report(self.output)


if __name__ == '__main__':
    unittest.main()
