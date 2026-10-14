"""Frozen consumer screen scope, matched baseline and safe predecessor gate."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.proxy_consumer_queue import CASES, make, predecessor_reference
from scripts.proxy_confirmation_queue import checkpoint_path, summarize, evaluate
from scripts.proxy_screen_evaluation import validate_scope


class QueueTests(unittest.TestCase):
    def test_recipe_and_all_smokes_before_production(self):
        from scripts.proxy_12layer_queue import ALL_SCREEN_ARMS
        self.assertEqual(len(ALL_SCREEN_ARMS),24)
        with tempfile.TemporaryDirectory() as tmp:
            jobs=make('proxy_heads.12layer.b200.json',Path(tmp)/'jobs.json','/rows','/reference')
            self.assertEqual(len(jobs),21)
            full=[j for j in jobs if j['name'].endswith('-train') and not j['name'].startswith('smoke-')]
            self.assertEqual(len(full),4)
            for case,j in zip(CASES,full):
                label,arm,weight,seed=case
                def arg(k):return j['argv'][j['argv'].index('--'+k)+1]
                for k,v in dict(arm=arm,seed=seed,data_seed=seed,proxy_module_seed=43,
                    proxy_lambda_max=weight,max_steps=2500,stop_after=2500,warmup_steps=125,
                    per_device_train_batch_size=16,gradient_accumulation_steps=4,block_size=2048,
                    num_hidden_layers=12,attention_backend='fa4',save_steps=250,save_total_limit=0).items():
                    self.assertEqual(arg(k),str(v),(arm,k))
                self.assertEqual(j['gpus'],list(range(8)))
                self.assertNotIn('--resume_from_checkpoint',j['argv'])
            self.assertLess(max(i for i,j in enumerate(jobs) if j['name'].startswith('smoke-')),jobs.index(full[0]))
            for j in jobs[-4:]:self.assertIn('scripts.proxy_consumer_queue',j['argv'])
            self.assertEqual([j['name'] for j in jobs[-3:]],['eval-numerics','eval-smoke','eval-full'])

    def test_scope_and_matched_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);reference=root/'reference';baseline=root/'original/supervised/run/seed-42/A'
            def put(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            put(reference/'scope.json',dict(source_screen=str(root/'original')))
            def result(arm,loss):return dict(status='complete',global_step=2500,arm=arm,
                evaluation=dict(eval_lm_loss=loss,eval_target_tokens=100),training_cost={})
            def config(arm,weight):return dict(pilot=dict(arm=arm,proxy_lambda_max=weight,proxy_module_seed=43,attention_backend='fa4'),
                training=dict(seed=42,data_seed=42,bf16=True),model_config=dict(num_hidden_layers=12),
                train_fingerprint='train',eval_fingerprint='eval',model=dict(tokenizer_name='/model'))
            put(baseline/'result.json',result('A',3.5));put(baseline/'train_config.json',config('A',.1))
            scope=dict(kind='consumer4',training_root=str(root),checkpoints=[],specs={})
            for case in CASES:
                label,arm,weight,seed=case;ck=checkpoint_path(root,case)
                put(ck.parent/'result.json',result(arm,3.6));put(ck.parent/'train_config.json',config(arm,weight))
                put(ck/'trainer_state.json',dict(global_step=2500));put(root/label/'validated.json',dict(status='passed'))
                scope['checkpoints'].append(str(ck));scope['specs'][str(ck)]=dict(label=label,arm=arm,lambda_max=weight,seed=seed,step=2500)
            validate_scope(scope);summarize(root,reference,cases=CASES)
            for r in json.loads((root/'summary.json').read_text())['results']:
                self.assertAlmostEqual(r['lm_loss_delta_vs_same_seed_A'],.1)
            bad=copy.deepcopy(scope);bad['specs'][bad['checkpoints'][-1]]['lambda_max']=.1
            with self.assertRaises(ValueError):validate_scope(bad)

    def test_predecessor_requires_success_and_burn_release(self):
        from scripts.proxy_confirmation_queue import CASES as previous
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sub=root/'supervised';guard=root/'guard'
            def put(name,value):
                p=sub/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
            burn=dict(session='burn',collective_progress_verified=True)
            scope=dict(kind='confirmation3',training_root='/previous',reference='/reference',checkpoints=[],specs={})
            for c in previous:
                label,arm,weight,seed=c;ck=str(checkpoint_path('/previous',c));scope['checkpoints'].append(ck)
                scope['specs'][ck]=dict(label=label,arm=arm,lambda_max=weight,seed=seed,step=2500)
            put('supervisor.json',dict(training_status='ok',training_returncode=0,finished_at='now',burn=burn))
            put('burn-verified.json',burn);put('run/run.json',dict(status='ok',jobs=[dict(status='ok')]*17))
            put('run/downstream-scope.json',scope)
            put('run/downstream/full/summary.json',dict(status='passed',diagnostic=False,checkpoints=dict.fromkeys(scope['checkpoints'])))
            self.assertEqual(predecessor_reference(root,guard),'/reference')
            guard.touch()
            with self.assertRaises(ValueError):predecessor_reference(root,guard)
            guard.unlink();burn['collective_progress_verified']=False;put('burn-verified.json',burn)
            with self.assertRaises(ValueError):predecessor_reference(root,guard)


if __name__=='__main__':unittest.main()
