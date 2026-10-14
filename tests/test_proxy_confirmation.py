"""Frozen mixed-seed queue, matched comparisons and evaluation handoff."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.proxy_confirmation_queue import CASES, EVAL_PYTHON, checkpoint_path, make, summarize
from scripts.proxy_screen_evaluation import validate_scope, predecessor_ready
from scripts.proxy_12layer_queue import ALL_SCREEN_ARMS
from run_experiments import load_jobs


class ConfirmationTests(unittest.TestCase):
    def test_queue_seeds_recipe_gates_and_evaluation(self):
        self.assertEqual(len(ALL_SCREEN_ARMS),24)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            items=make('proxy_heads.12layer.b200.json',root/'jobs.json',root/'rows.json','/previous')
            self.assertEqual(len(load_jobs(root/'jobs.json')),17)
            self.assertEqual(items[0]['name'],'numerics')
            full=[j for j in items if j['name'].endswith('-train') and not j['name'].startswith('smoke-')]
            self.assertEqual(len(full),3)
            for case,j in zip(CASES,full):
                label,arm,weight,seed=case
                def arg(k):return j['argv'][j['argv'].index('--'+k)+1]
                for k,v in dict(arm=arm,seed=seed,data_seed=seed,max_steps=2500,stop_after=2500,
                               warmup_steps=125,num_hidden_layers=12,proxy_lambda_max=weight,
                               save_total_limit=0,save_steps=250,per_device_train_batch_size=16,
                               gradient_accumulation_steps=4,block_size=2048,attention_backend='fa4').items():
                    self.assertEqual(arg(k),str(v),(arm,k))
                self.assertIn(f'PYTHONHASHSEED={seed}',j['argv'])
                if arm!='A':self.assertEqual(arg('proxy_module_seed'),str(seed+1))
                self.assertEqual(arg('output_dir'),str(checkpoint_path('{run_dir}',case).parent))
                self.assertNotIn('--resume_from_checkpoint',j['argv'])
                self.assertEqual(j['gpus'],list(range(8)))
                validator=items[items.index(j)+1]
                self.assertEqual(validator['argv'][validator['argv'].index('--seed')+1],str(seed))
            for j in items[-3:]:
                self.assertEqual(j['argv'][0],EVAL_PYTHON)
                self.assertEqual(j['gpus'],list(range(8)))
            self.assertEqual([j['name'] for j in items[-4:]],['summarize','eval-numerics','eval-smoke','eval-full'])
            outputs=[r['path'] for j in items for r in j['required_outputs']]
            self.assertEqual(len(outputs),len(set(outputs)))
            self.assertLess(max(i for i,j in enumerate(items) if j['name'].startswith('smoke-')),items.index(full[0]))

    def test_scope_rejects_wrong_seed_or_checkpoint(self):
        scope=dict(kind='confirmation3',training_root='/run',checkpoints=[],specs={})
        for case in CASES:
            label,arm,weight,seed=case;ck=str(checkpoint_path('/run',case))
            scope['checkpoints'].append(ck)
            scope['specs'][ck]=dict(label=label,arm=arm,lambda_max=weight,seed=seed,step=2500)
        validate_scope(scope)
        scope['specs'][scope['checkpoints'][1]]['seed']=42
        with self.assertRaises(ValueError):validate_scope(scope)

    def test_summary_compares_only_matching_seed_and_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';reference=Path(tmp)/'previous';reference.mkdir()
            original=Path(tmp)/'original';base=original/'supervised/run/seed-42/A';base.mkdir(parents=True)
            (reference/'scope.json').write_text(json.dumps(dict(source_screen=str(original))))
            def write(folder,name,value):
                folder.mkdir(parents=True,exist_ok=True);(folder/name).write_text(json.dumps(value))
            def result(arm,loss):return dict(arm=arm,global_step=2500,status='complete',training_cost={},
                evaluation=dict(eval_lm_loss=loss,eval_target_tokens=100))
            def config(arm,seed):return dict(pilot=dict(arm=arm,proxy_lambda_max=.1,proxy_module_seed=seed+1,attention_backend='fa4'),
                model_config=dict(num_hidden_layers=12),training=dict(seed=seed,data_seed=seed,bf16=True),
                train_fingerprint=f'train-{seed}',eval_fingerprint='eval',model=dict(tokenizer_name='/tokenizer'))
            write(base,'result.json',result('A',3.5));write(base,'train_config.json',config('A',42))
            for case,loss in zip(CASES,(3.6,3.58,3.48)):
                label,arm,weight,seed=case;ck=checkpoint_path(root,case)
                write(ck.parent,'result.json',result(arm,loss));write(ck.parent,'train_config.json',config(arm,seed))
                write(ck,'trainer_state.json',dict(global_step=2500));write(root/label,'validated.json',dict(status='passed'))
            summarize(root,reference)
            report=json.loads((root/'summary.json').read_text())
            for row,want in zip(report['results'],(0.,-.02,-.02)):
                self.assertAlmostEqual(row['lm_loss_delta_vs_same_seed_A'],want)
            (root/'summary.json').unlink()
            ck=checkpoint_path(root,CASES[1]);bad=config(CASES[1][1],1042);bad['train_fingerprint']='wrong'
            write(ck.parent,'train_config.json',bad)
            with self.assertRaises(ValueError):summarize(root,reference)
            self.assertFalse((root/'summary.json').exists())

    def test_completed_27_evaluation_and_burn_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sub=root/'supervised';(sub/'run/full').mkdir(parents=True);guard=root/'guard'
            def put(n,x):(sub/n).write_text(json.dumps(x))
            burn=dict(session='burn',collective_progress_verified=True)
            put('supervisor.json',dict(training_status='ok',training_returncode=0,finished_at='now',burn=burn))
            put('burn-verified.json',burn);put('run/complete.json',{})
            put('run/run.json',dict(status='ok',jobs=[dict(name=n,status='ok') for n in ('numerics','smoke','full')]))
            put('run/full/summary.json',dict(status='passed',diagnostic=False,checkpoints={str(i):{} for i in range(27)}))
            self.assertTrue(predecessor_ready(root,guard,evaluation=True,expected_count=27))
            guard.touch();self.assertFalse(predecessor_ready(root,guard,evaluation=True,expected_count=27))
            guard.unlink();put('supervisor.json',dict(training_status='failed'))
            with self.assertRaises(RuntimeError):predecessor_ready(root,guard,evaluation=True,expected_count=27)

    def test_post_training_eval_uses_frozen_three_checkpoint_scope(self):
        from scripts.proxy_confirmation_queue import evaluate
        from scripts.proxy_screen_evaluation import PROTOCOLS
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'summary.json').write_text(json.dumps(dict(status='passed')))
            cache=root/'accelerate.yaml';cache.write_bytes(Path('resources/accelerate_config.yaml').read_bytes())
            rows=[dict(label=label,arm=arm,lambda_max=weight,seed=seed,checkpoint=str(checkpoint_path(root,case)))
                  for case in CASES for label,arm,weight,seed in (case,)]
            previous=dict(protocols={k:[n,t] for k,(n,t) in PROTOCOLS.items()},data_root='/data',
                          manifest='/manifest',manifest_sha256='data-hash',versions={})
            with patch('scripts.proxy_confirmation_queue.evaluation_reference',return_value=previous), \
                 patch('scripts.proxy_confirmation_queue.validate_training',return_value=rows), \
                 patch('scripts.train_then_burn.digest',return_value='weight-hash'), \
                 patch('accelerate.commands.config.config_args.default_yaml_config_file',str(cache)), \
                 patch('subprocess.run') as command, patch('scripts.proxy_screen_evaluation.run') as run:
                evaluate(root,'/reference','numerics')
                scope=json.loads((root/'downstream-scope.json').read_text());validate_scope(scope)
                self.assertEqual(scope['evaluation_seed'],42)
                self.assertEqual([scope['specs'][c]['seed'] for c in scope['checkpoints']],[1042,1042,42])
                self.assertEqual(command.call_args.args[0][1],'env')
                run.assert_called_with('numerics',root/'downstream/numerics',root/'downstream-scope.json')
                evaluate(root,'/reference','smoke');evaluate(root,'/reference','full')
                self.assertEqual(run.call_count,3)
                self.assertEqual(json.loads((root/'downstream-scope.json').read_text()),scope)


if __name__=='__main__':unittest.main()
