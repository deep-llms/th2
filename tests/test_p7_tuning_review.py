"""Regression checks for exact numerical recipes and cross-case launch gates."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from scripts.check_fa4_proxy import numerical_settings, run
from scripts.p7_tuning_queue import CASES, make, check_case_config, verify
from tests.test_p7_tuning import config
from tests.test_proxy_screen24 import fixture
from tests.test_train import invoke


class TuningReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_exact_recipe_and_legacy_defaults(self):
        recipe=dict(arm='P7-simple-targetgrad',seed=2042,data_seed=2042,
            proxy_module_seed=2043,proxy_lambda_max=.3,proxy_width=32,
            proxy_compile_auxiliary=True,proxy_compile_entries=True,
            proxy_fused_targets=True,proxy_lse_attention=True)
        settings,seed=numerical_settings(recipe['arm'],recipe,True)
        self.assertEqual((settings.lambda_max,settings.width,settings.module_seed,seed),(.3,32,2043,2042))
        self.assertTrue(settings.fused_targets and settings.lse_attention)
        legacy,seed=numerical_settings(recipe['arm'],recipe,False)
        self.assertEqual((legacy.lambda_max,legacy.width,legacy.module_seed,seed),(.1,256,43,42))
        self.assertFalse(legacy.fused_targets)
        with self.assertRaises(ValueError):numerical_settings('P7-simple-lm',recipe,True)
        with self.assertRaises(ValueError):numerical_settings(recipe['arm'],{**recipe,'data_seed':42},True)

    def test_repeated_arm_cases_and_recipe_hashes(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            recipes=[]
            for i,weight in enumerate((.03,.3)):
                path=root/f'{i}.json';path.write_text(json.dumps(dict(seed=42,proxy_lambda_max=weight)))
                recipes.append(dict(label=f'case-{i}',arm='P7-simple-targetgrad',recipe=str(path)))
            manifest=root/'manifest.json';manifest.write_text(json.dumps(recipes))
            commands=[]
            class Child:
                def __init__(self,command,env,**kw):
                    commands.append((command,env))
                    recipe=Path(command[command.index('--recipe')+1])
                    output=Path(command[command.index('--output')+1])
                    output.write_text(json.dumps(dict(status='passed',arm='P7-simple-targetgrad',
                        exact_recipe=True,recipe_sha256=hashlib.sha256(recipe.read_bytes()).hexdigest())))
                def wait(self,**kw):return 0
                def poll(self):return 0
            args=SimpleNamespace(case_manifest=str(manifest),worker_arms=None,rows='rows',output=str(root/'out'))
            with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'),patch('scripts.check_fa4_proxy.subprocess.Popen',Child):
                run(args)
            self.assertEqual(len(commands),2)
            for (command,env),recipe in zip(commands,recipes):
                self.assertIn('--exact-recipe',command)
                self.assertIn(recipe['recipe'],command);self.assertEqual(env['PYTHONHASHSEED'],'42')
            result=json.loads((root/'out/summary.json').read_text())
            self.assertEqual([r['case_label'] for r in result['cases']],['case-0','case-1'])
            class WrongHash(Child):
                def __init__(self,command,**kw):
                    super().__init__(command,**kw)
                    output=Path(command[command.index('--output')+1]);value=json.loads(output.read_text())
                    value['recipe_sha256']='wrong';output.write_text(json.dumps(value))
            args.output=str(root/'bad')
            with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'),patch('scripts.check_fa4_proxy.subprocess.Popen',WrongHash):
                with self.assertRaises(AssertionError):run(args)

    def test_real_resolved_recipes_and_guard_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=fixture(root);config().save_pretrained(root/'model')
            # Actual train.py artifacts, both a baseline and an architecture
            # variant, plus the changed seed used by the replication pair.
            examples=[]
            for label,arm,weight,seed in (('a-seed2042','A',.1,2042),
                    ('lm-noaux-seed2042','P7-simple-lm-noaux',0.,2042),
                    ('lm-valueonly-gated','P7-simple-lm-valueonly-gated',.1,42)):
                r={**base,'arm':arm,'seed':seed,'data_seed':seed,'proxy_lambda_max':weight,
                   'proxy_module_seed':None if arm=='A' else seed+1,'output_dir':str(root/label)}
                invoke(root,r)
                c=json.loads((root/label/'train_config.json').read_text())
                check_case_config(c,r)
                for section,key,value in [('training','learning_rate',.05),('training','seed',99),
                    ('pilot','proxy_lambda_max',1.5),('pilot','proxy_sync_optimization',False),
                    ('data','data_dir','wrong')]:
                    bad=copy.deepcopy(c);bad[section][key]=value
                    with self.assertRaises(ValueError,msg=key):check_case_config(bad,r)
                examples.append((label,c))
            self.assertEqual(examples[0][1]['train_fingerprint'],examples[1][1]['train_fingerprint'])
            self.assertNotEqual(examples[0][1]['train_fingerprint'],examples[2][1]['train_fingerprint'])
            self.assertEqual(examples[0][1]['eval_fingerprint'],examples[2][1]['eval_fingerprint'])
            from deep_kv.report import comparable_config
            normalized=[]
            for _,c in examples:
                common=comparable_config(c);common.pop('train_fingerprint')
                common['training'].update(seed=42,data_seed=42)
                common['pilot']['proxy_lambda_max']=.1
                normalized.append(common)
            self.assertEqual(normalized[0],normalized[1])
            self.assertEqual(normalized[0],normalized[2])

    def test_same_seed_fingerprint_and_retention_gate(self):
        # Real model/report semantics are covered above and in test_p7_tuning;
        # this fixture exercises the queue's additional aggregate invariants.
        from dataclasses import asdict
        from deep_kv.proxy import resolve_proxy_settings, ProxySettings
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);labels=['tg-short','tg-medium']
            make('proxy_heads.12layer.b200.json',root/'jobs.json','rows',labels)
            for label,arm,weight,seed in (c for c in CASES if c[0] in labels):
                folder=root/'run'/label/f'seed-{seed}'/arm;folder.mkdir(parents=True)
                requested=json.loads((root/'jobs.config'/f'{label}.json').read_text())
                opts={k.removeprefix('proxy_'):v for k,v in requested.items() if k.startswith('proxy_') and k.removeprefix('proxy_') in ProxySettings.__dataclass_fields__}
                settings=resolve_proxy_settings(arm,ProxySettings(**opts))
                pilot={**{'proxy_'+k:v for k,v in asdict(settings).items()},'arm':arm,'proxy_screen':True}
                c=dict(model={},model_config={},data={},pilot=pilot,training=dict(seed=seed,data_seed=seed),
                       train_fingerprint='train42',eval_fingerprint='eval')
                (folder/'train_config.json').write_text(json.dumps(c))
                (folder/'result.json').write_text(json.dumps(dict(evaluation={})))
                (folder.parent.parent/'validated.json').write_text(json.dumps(dict(status='passed')))
                for step in range(250,2501,250):(folder/f'checkpoint-{step}').mkdir()
            def check():
                with patch('scripts.p7_tuning_queue.check_case_config'),patch('deep_kv.report.report'):
                    return verify(root/'run',root/'jobs.config',labels)
            self.assertEqual(check()['status'],'passed');(root/'run/summary.json').unlink()
            folder=root/'run/tg-medium/seed-42/P7-simple-targetgrad-medium'
            c=json.loads((folder/'train_config.json').read_text());c['train_fingerprint']='wrong'
            (folder/'train_config.json').write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError,'fingerprints'):check()
            c['train_fingerprint']='train42';(folder/'train_config.json').write_text(json.dumps(c))
            (folder/'checkpoint-250').rmdir()
            with self.assertRaisesRegex(ValueError,'retention'):check()


if __name__=='__main__':unittest.main()
