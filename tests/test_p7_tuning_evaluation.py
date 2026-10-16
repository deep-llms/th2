import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.p7_tuning_evaluation import selected_cases, build_scope, validate_scope
from scripts.p7_tuning_queue import ALL_CASES
from scripts.proxy_confirmation_queue import checkpoint_path
from scripts.proxy_screen_evaluation import validate_scope as shared_validate, PROTOCOLS


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class TuningEvaluationTests(unittest.TestCase):
    def selection(self, root, cases):
        jobs = []
        for c in cases:
            write(root/c[0]/'validated.json', dict(status='passed'))
            jobs += [dict(name=c[0]+suffix, status='ok') for suffix in ('-train', '-validate')]
        write(root/'run.json', dict(status='failed', jobs=jobs))

    def test_selects_across_roots_and_excludes_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'a', Path(tmp)/'b'
            self.selection(a, ALL_CASES[:2]); self.selection(b, ALL_CASES[5:7])
            checkpoint_path(b, ALL_CASES[7]).mkdir(parents=True)
            self.assertEqual([c for _, c in selected_cases([a,b])], list(ALL_CASES[:2]+ALL_CASES[5:7]))
            write(b/'run.json', dict(status='running',jobs=[]))
            with self.assertRaisesRegex(ValueError,'Stop training'): selected_cases([a,b])

    def test_rejects_duplicate_or_unvalidated_queue_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'a', Path(tmp)/'b'
            self.selection(a, ALL_CASES[:1]); self.selection(b, ALL_CASES[:1])
            with self.assertRaisesRegex(ValueError,'Duplicate'): selected_cases([a,b])
            write(b/'run.json',dict(status='failed',jobs=[]))
            with self.assertRaisesRegex(ValueError,'queue receipt'): selected_cases([b])

    def test_scope_preserves_seeds_controls_recipes_and_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'training'; reference=root/'reference'
            cases=(ALL_CASES[0],)+tuple(c for c in ALL_CASES if c[3]==2042)
            self.selection(source,cases)
            config_root=root/'configs'
            write(source/'jobs.snapshot.json',dict(jobs=[dict(name='summary',argv=['--config-root',str(config_root)])]))
            items=[(checkpoint_path(source,c),*c) for c in cases]
            items += [(reference/'supervised/run/seed-42'/arm/'checkpoint-2500',label,arm,.1,42)
                      for label,arm in [('reference-A','A'),('reference-lm','P7-simple-lm')]]
            for ck,label,arm,weight,seed in items:
                c=dict(training=dict(max_steps=2500,warmup_steps=125,seed=seed,data_seed=seed,bf16=True),
                       model_config=dict(num_hidden_layers=12),model=dict(tokenizer_name='tok'),
                       pilot=dict(arm=arm,proxy_lambda_max=weight,proxy_module_seed=seed+1,attention_backend='fa4'),
                       train_fingerprint='train-'+str(seed),eval_fingerprint='same-eval')
                write(ck.parent/'train_config.json',c)
                write(ck.parent/'result.json',dict(status='complete',global_step=2500,arm=arm))
                write(ck/'trainer_state.json',dict(global_step=2500))
                (ck/'model.safetensors').write_bytes(b'weights')
                write(config_root/(label+'.json'),{})
            previous=dict(source_screen=str(reference),protocols={k:[n,t] for k,(n,t) in PROTOCOLS.items()},
                          data_root='/data',manifest='/manifest',manifest_sha256='hash',versions={})
            with patch('scripts.p7_tuning_evaluation.evaluation_reference',return_value=previous), \
                 patch('scripts.p7_tuning_evaluation.validate_training',return_value=[]), \
                 patch('scripts.p7_tuning_evaluation.check_case_config') as recipes, \
                 patch('deep_kv.report.report'):
                scope,_=build_scope([source],'/reference-eval')
                self.assertEqual(len(scope['checkpoints']),5)
                self.assertEqual(recipes.call_count,3)
                shared_validate(scope)
                for key,value in [('seed',42),('step',2000),('lambda_max',.3),('arm','wrong')]:
                    bad=copy.deepcopy(scope);bad['specs'][scope['checkpoints'][1]][key]=value
                    with self.assertRaises(ValueError):validate_scope(bad)
                p=items[-1][0].parent/'train_config.json';c=json.loads(p.read_text())
                c['train_fingerprint']='changed';write(p,c)
                with self.assertRaisesRegex(ValueError,'data/tokenizer'):build_scope([source],'/reference-eval')


if __name__=='__main__':unittest.main()
