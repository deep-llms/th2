import copy
import json
from pathlib import Path
import tempfile
import unittest

from deep_kv.__main__ import jobs
from scripts.proxy_time_match import make, report, timing_plan


class TimeMatchTests(unittest.TestCase):
    def test_plan_preserves_schedule_and_generates_exact_resume_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source'; output=root/'new'
            def write(path,value):
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(value))
            results=[]
            for arm,runtime in [('A',21059.0039),('P6-iso',22743.7338)]:
                r=dict(arm=arm,global_step=10000,schedule_steps=28600,
                    training_cost=dict(optimizer_steps=10000,runtime_seconds=runtime),
                    evaluation=dict(eval_lm_loss=3.02))
                results.append(r);write(source/f'seed-1042/{arm}/result.json',r)
                write(source/f'seed-1042-validate-{arm}.json',dict(status='passed'))
            cfg=dict(training=dict(seed=1042,data_seed=1042,save_steps=250,save_total_limit=2,
                                   ignore_data_skip=False),world_size=8,train_fingerprint='a',
                                   eval_fingerprint='b',tokens_per_update=1048576)
            write(source/'seed-1042/A/train_config.json',cfg)
            write(source/'run.json',dict(status='ok'))
            original=jobs(Path(__file__).resolve().parents[1]/'proxy_heads.b200.json',
                          stop_after=10000,arms=('A',),seeds=[1042])
            argv=original['jobs'][0]['argv'];argv[argv.index('--save_total_limit')+1]='2'
            original_argv=argv.copy()
            write(source/'jobs.snapshot.json',original)
            plan=make(source,output)
            self.assertEqual(plan['stop_after'],10800)
            generated=json.loads((output/'jobs.json').read_text())['jobs']
            self.assertEqual(len(generated),4)
            resumed=generated[1]['argv']
            self.assertEqual(resumed[-2:],['--resume_from_checkpoint','{run_dir}/seed-1042/A/checkpoint-10000'])
            restored=resumed[:-2].copy()
            for flag in ('--save_total_limit','--stop_after'):
                restored[restored.index(flag)+1]=original_argv[original_argv.index(flag)+1]
            self.assertEqual(restored,original_argv)
            # The final report uses cumulative runtime, not the resumed fragment alone.
            final=root/'finished'; current=final/'seed-1042/A'
            resumed_result=copy.deepcopy(results[0]);resumed_result.update(global_step=10800)
            resumed_result['training_cost']=dict(optimizer_steps=800,runtime_seconds=1685)
            write(current/'result.json',resumed_result)
            cfg['training']['save_total_limit']=0;write(current/'train_config.json',cfg)
            write(current/'resume-transition-test.json',dict(changes={'save_total_limit':dict(before=2,after=0)}))
            for step in (10000,10250,10500,10750,10800): (current/f'checkpoint-{step}').mkdir()
            value=report(source,final)
            self.assertTrue(value['within_one_percent'])
            self.assertAlmostEqual(value['results'][-1]['runtime_seconds'],22744.0039)
            self.assertEqual(value['retained_checkpoints'],[10000,10250,10500,10750,10800])
            bad=copy.deepcopy(results[1]);bad['training_cost']['runtime_seconds']=float('nan')
            with self.assertRaises(AssertionError):timing_plan(results[0],bad)


if __name__=='__main__': unittest.main()
