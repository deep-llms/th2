"""Completed-checkpoint selection and unchanged downstream evaluation protocol."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.p7_exploration_queue import CASES, checkpoint_path
from scripts.p7_completed_evaluation import selected_cases, build_scope, validate_scope
from scripts.proxy_screen_evaluation import PROTOCOLS, make_jobs, validate_scope as shared_validate
from run_experiments import load_jobs


class CompletedEvaluationTests(unittest.TestCase):
    def test_selection_requires_keyonly_and_skips_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def complete(case):
                folder = checkpoint_path(root, case).parent
                folder.mkdir(parents=True)
                (folder / 'result.json').write_text('{}')
                (root / case[0] / 'validated.json').write_text('{}')
            for case in CASES[:4]: complete(case)
            with self.assertRaises(ValueError): selected_cases(root)
            complete(CASES[4])
            checkpoint_path(root, CASES[5]).mkdir(parents=True)
            self.assertEqual(selected_cases(root), CASES[:5])
            complete(CASES[6])
            with self.assertRaises(ValueError): selected_cases(root)

    def fixture(self, root):
        source = root / 'source'; reference = root / 'reference'
        configs = []
        items = [(checkpoint_path(source, c), c[1], c[2], c[3]) for c in CASES[:6]]
        items += [(reference / 'supervised/run/seed-42' / arm / 'checkpoint-2500', arm, 2500, .1)
                  for arm in ('A', 'P7-simple-lm')]
        for ck, arm, steps, weight in items:
            ck.mkdir(parents=True)
            c = dict(training=dict(max_steps=steps, warmup_steps=steps // 20, seed=42, data_seed=42, bf16=True),
                     model_config=dict(num_hidden_layers=12), model=dict(tokenizer_name='tokenizer'),
                     pilot=dict(proxy_lambda_max=weight), train_fingerprint='train', eval_fingerprint='eval')
            (ck.parent / 'train_config.json').write_text(json.dumps(c)); configs.append(ck.parent / 'train_config.json')
            (ck.parent / 'result.json').write_text(json.dumps(dict(status='complete', global_step=steps)))
            (ck / 'trainer_state.json').write_text(json.dumps(dict(global_step=steps)))
            (ck / 'model.safetensors').write_bytes(b'test-weights')
        for c in CASES[:6]: (source / c[0] / 'validated.json').write_text('{"status":"passed"}')
        previous = dict(source_screen=str(reference), protocols={k:[n,t] for k,(n,t) in PROTOCOLS.items()},
                        data_root='/data', manifest='/manifest', manifest_sha256='hash', versions={})
        return source, previous, configs

    def test_scope_controls_and_recipe_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, previous, configs = self.fixture(Path(tmp))
            with patch('scripts.p7_completed_evaluation.evaluation_reference', return_value=previous), \
                 patch('scripts.p7_completed_evaluation.validate_training', return_value=[]) as validate, \
                 patch('deep_kv.report.report'):
                scope, _ = build_scope(source, '/reference-eval')
                validate.assert_called_once_with(source, cases=CASES[:6])
                self.assertEqual(len(scope['checkpoints']), 8)
                validate_scope(scope); shared_validate(scope)
                for key, value in [('step', 10000), ('seed', 1042), ('lambda_max', .3), ('arm', 'wrong')]:
                    bad = copy.deepcopy(scope); bad['specs'][bad['checkpoints'][-1]][key] = value
                    with self.assertRaises(ValueError): shared_validate(bad)
                for labels in (['lm-keyonly'], scope['completed_labels'] * 2, list(reversed(scope['completed_labels']))):
                    with self.assertRaises(ValueError): validate_scope({**scope, 'completed_labels': labels})
                c = json.loads(configs[-1].read_text()); c['train_fingerprint'] = 'changed'
                configs[-1].write_text(json.dumps(c))
                with self.assertRaisesRegex(ValueError, 'data/tokenizer'): build_scope(source, '/reference-eval')

    def test_existing_eight_gpu_numerical_smoke_full_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / 'jobs.json'; jobs = make_jobs(root / 'scope.json')
            p.write_text(json.dumps(dict(jobs=jobs)))
            load_jobs(p)
            self.assertEqual([j['name'] for j in jobs], ['numerics', 'smoke', 'full'])
            self.assertTrue(all(j['gpus'] == list(range(8)) for j in jobs))
            self.assertTrue(all('scripts.proxy_screen_evaluation' in j['argv'] for j in jobs))


if __name__ == '__main__': unittest.main()
