"""Frozen offline plan, safe outputs and mixed-horizon evaluation scope."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.p7_exploration_queue import CASES, OPTIMIZATIONS, checkpoint_path, make, validate_scope
from run_experiments import load_jobs


class ExplorationQueueTests(unittest.TestCase):
    def test_all_gates_precede_fresh_runs_and_mixed_schedules(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'jobs.json'
            items=make('proxy_heads.12layer.b200.json',path,'/rows.json','/previous')
            self.assertEqual(len(load_jobs(path)),38)
            self.assertEqual(items[0]['name'],'numerics')
            full=[j for j in items if j['name'].endswith('-train') and not j['name'].startswith('smoke-')]
            smokes=[j for j in items if j['name'].startswith('smoke-')]
            self.assertEqual(len(full),8);self.assertEqual(len(smokes),16)
            self.assertLess(max(items.index(j) for j in smokes),items.index(full[0]))
            self.assertEqual(items[items.index(full[0])-1]['name'],'verify-smoke-recipes')
            for (label,arm,steps,weight),job in zip(CASES,full):
                def arg(k):return job['argv'][job['argv'].index('--'+k)+1]
                for k,v in dict(arm=arm,max_steps=steps,stop_after=steps,warmup_steps=steps//20,
                    proxy_lambda_max=weight,seed=42,data_seed=42,num_hidden_layers=12,
                    block_size=2048,per_device_train_batch_size=16,gradient_accumulation_steps=4,
                    save_steps=250,save_total_limit=0,attention_backend='fa4').items():
                    self.assertEqual(arg(k),str(v),(arm,k))
                self.assertEqual(arg('output_dir'),str(checkpoint_path('{run_dir}',(label,arm,steps,weight)).parent))
                self.assertNotIn('--resume_from_checkpoint',job['argv'])
                self.assertEqual(job['gpus'],list(range(8)))
                for key in OPTIMIZATIONS:self.assertEqual('--'+key in job['argv'],arm=='P7-simple-targetgrad')
                check=items[items.index(job)+1]
                self.assertEqual(check['argv'][check['argv'].index('--schedule-steps')+1],str(steps))
            self.assertEqual([j['name'] for j in items[-4:]],['summarize','eval-numerics','eval-smoke','eval-full'])
            self.assertNotIn('__main__',[a for j in items for a in j['argv']])
            outputs=[r['path'] for j in items for r in j['required_outputs']]
            self.assertEqual(len(outputs),len(set(outputs)))
            old=path.read_bytes()
            with self.assertRaises(FileExistsError):make('proxy_heads.12layer.b200.json',path,'/rows','/previous')
            self.assertEqual(old,path.read_bytes())

    def scope(self):
        specs={str(checkpoint_path('/run',case)):dict(label=case[0],arm=case[1],step=case[2],lambda_max=case[3],seed=42,checkpoint_sha256='hash') for case in CASES}
        return dict(kind='p7-exploration8',training_root='/run',checkpoints=list(specs),specs=specs)

    def test_scope_rejects_changed_seed_horizon_or_order(self):
        from scripts.proxy_screen_evaluation import validate_scope as shared
        scope=self.scope();validate_scope(scope);shared(scope)
        for key,value in (('seed',1042),('step',10000),('lambda_max',.3),('arm','P7-simple')):
            changed=copy.deepcopy(scope);changed['specs'][changed['checkpoints'][0]][key]=value
            with self.assertRaises(ValueError):shared(changed)
        scope['checkpoints'].reverse()
        with self.assertRaises(ValueError):shared(scope)

    def test_numerical_eval_uses_each_actual_checkpoint_step(self):
        from scripts.proxy_screen_evaluation import run,PROTOCOLS
        scope=self.scope();scope.update(protocols={k:[n,t] for k,(n,t) in PROTOCOLS.items()},manifest='/manifest',manifest_sha256='hash')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'scope.json';path.write_text(json.dumps(scope))
            def execute(jobs,gpus):
                self.assertEqual(gpus,8)
                for job,case in zip(jobs,CASES):
                    argv=job['argv'];self.assertEqual(argv[argv.index('--expected-step')+1],str(case[2]))
                    Path(job['output']).write_text(json.dumps(dict(status='passed',arm=case[1],step=case[2],checkpoint_sha256='hash')))
            with patch('eval.models.file_hash',return_value='hash'),patch('eval.suite.execute',side_effect=execute):
                run('numerics',root/'numerics',path)
            self.assertEqual(json.loads((root/'numerics/summary.json').read_text())['status'],'passed')

    def test_smoke_gate_rejects_data_or_recipe_change_before_training(self):
        from scripts.p7_exploration_queue import verify_smokes
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';reference=Path(tmp)/'reference';original=Path(tmp)/'original'
            def put(path,value):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            base=dict(pilot=dict(arm='A',proxy_screen=True,attention_backend='fa4',
                      proxy_lambda_max=.1,proxy_groups=2,proxy_lookahead=4),
                training=dict(max_steps=2500,warmup_steps=125,seed=42,data_seed=42,learning_rate=.0003,
                              logging_steps=10,save_steps=250,eval_steps=512,eval_on_start=True),
                data=dict(eval_rows=4882,monitor_rows=128,isolate_documents=True),
                train_fingerprint='train-original',eval_fingerprint='eval-original')
            put(reference/'scope.json',dict(source_screen=str(original)))
            put(original/'supervised/run/seed-42/A/train_config.json',base)
            put(original/'supervised/run/seed-42/A/result.json',dict(global_step=2500,status='complete'))
            for case in CASES:
                label,arm,steps,weight=case;c=copy.deepcopy(base)
                c['pilot'].update(arm=arm,proxy_lambda_max=weight)
                if arm!='A':c['pilot'].update(proxy_target_version='p4p6-r1',proxy_loss_form='cosine',
                                             proxy_isolate_estimator=True,proxy_module_seed=43)
                if arm=='P7-simple-targetgrad':c['pilot'].update({k:True for k in OPTIMIZATIONS})
                c['training'].update(max_steps=steps,warmup_steps=steps//20,save_steps=3,
                                     logging_steps=1,eval_steps=3,eval_on_start=False)
                c['data'].update(eval_rows=16,monitor_rows=16);c['eval_fingerprint']='eval-smoke'
                put(checkpoint_path(root/'smoke',case).parent/'train_config.json',c)
                put(root/'smoke'/label/'validated.json',dict(status='passed'))
            verify_smokes(root,reference)
            self.assertTrue((root/'smoke-recipes.json').exists());(root/'smoke-recipes.json').unlink()
            path=checkpoint_path(root/'smoke',CASES[-1]).parent/'train_config.json'
            good=json.loads(path.read_text())
            for key,value in (('train_fingerprint','different'),('learning_rate',.001)):
                bad=copy.deepcopy(good)
                if key=='learning_rate':bad['training'][key]=value
                else:bad[key]=value
                put(path,bad)
                with self.assertRaises(ValueError):verify_smokes(root,reference)
                self.assertFalse((root/'smoke-recipes.json').exists())


if __name__=='__main__':unittest.main()
