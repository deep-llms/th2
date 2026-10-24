import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts.stop_long_launch import prepare, FIT_START, STEPS, START_SEQUENCE
from eval.stop_v2_study import validate_stage
from run_experiments import load_jobs
from resume_gpu_probe import prepare as prepare_fixture, digest

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class LongStopTests(unittest.TestCase):
    def test_queue_preserves_schedule_and_uses_explicit_midpoint_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);parent = root/'parent';previous = root/'previous';run = root/'run'
            run.mkdir()
            base = dict(max_steps=10000, block_size=1024, seed=0, data_seed=0,
                        per_device_train_batch_size=16, gradient_accumulation_steps=4,
                        train_sampling_strategy='sequential', dataloader_drop_last=True,
                        attention_backend='fa4', gradient_checkpointing=False, save_steps=250,
                        save_total_limit=None, output_dir=str(parent/'original'),
                        lr_scheduler_type='cosine_with_min_lr', lr_scheduler_kwargs={'min_lr_rate': .1})
            write(parent/'prepared/D-s0.train.json', base)
            write(previous/'evaluation_manifest.json', {n: dict(checkpoint=str(previous/n/'checkpoint-1000'))
                  for n in ('D-continue','D-DualCE6')})
            with patch('scripts.stop_long_launch.initialize') as initializer:
                jobs = prepare(parent, previous, run, ROOT)
            self.assertEqual(len(initializer.call_args_list), 2)
            for call in initializer.call_args_list:
                self.assertEqual(call.args[0], parent/'original/checkpoint-10000')
                self.assertEqual(call.kwargs['schedule_steps'], 10000)
            self.assertEqual([j['name'] for j in jobs], ['capacity','resume_gate','eval_smoke',
                             'D-continue-5000','D-DualCE6-5000','eval_5000',
                             'D-continue-10000','D-DualCE6-10000','eval_10000'])
            load_jobs(run/'jobs.json')
            for name in ('D-continue','D-DualCE6'):
                a=json.loads((run/f'prepared/{name}-5000.train.json').read_text())
                b=json.loads((run/f'prepared/{name}-10000.train.json').read_text())
                self.assertEqual(a['stop_after'],5000);self.assertEqual(b['stop_after'],10000)
                self.assertEqual(b.pop('resume_from_checkpoint'),str(run/f'prepared/runs/{name}/checkpoint-5000'))
                a.pop('stop_after');b.pop('stop_after');self.assertEqual(a,b)
                self.assertEqual(a['max_steps'],10000);self.assertEqual(a['warmup_steps'],200)
                self.assertEqual(a['train_start_sequence'],5120000)
            self.assertEqual(FIT_START, START_SEQUENCE+STEPS*512)
            self.assertEqual(FIT_START,10240000)

    def test_calibration_excludes_future_training_and_wrong_stage_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);cp=run/'checkpoint-5000'
            write(cp/'trainer_state.json',dict(global_step=5000))
            write(cp/'train_config.json',dict(data=dict(train_start_sequence=5120000),
                  training=dict(max_steps=10000,per_device_train_batch_size=16,gradient_accumulation_steps=4)))
            write(cp/'training_recipe.json',dict(execution=dict(world_size=8)))
            write(run/'train_results.json',dict(global_step=5000))
            with self.assertRaisesRegex(ValueError,'overlaps'):
                validate_stage(cp,5000,7680000)  # beyond midpoint, but inside future training
            validate_stage(cp,5000,10240000)
            write(run/'train_results.json',dict(global_step=10000))
            with self.assertRaisesRegex(ValueError,'another stage'):
                validate_stage(cp,5000,10240000)

    def test_real_checked_stages_match_uninterrupted_dualce_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            config=json.loads((ROOT/'configs/bdot/D.json').read_text())
            config['bdot_stop']=dict(train_random_prob=0.,min_steps=6,exit_steps=[6])
            config['bdot_exit_training']=dict(depth=6,ce_weight=.5,kl_weight=0.)
            write(root/'variant.json',config)
            prepare_fixture(root/'fixture','sdpa',1,4,2,1,True,curriculum='none',
                            variant_config=root/'variant.json')
            base=json.loads((root/'fixture/full.json').read_text())
            base.pop('keep_checkpoint_steps',None)
            base['output_dir']=str(root/'stage')
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1',
                 'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','WANDB_MODE':'offline'}

            def execute(argv,label):
                result=subprocess.run(argv,cwd=ROOT,env=env,text=True,capture_output=True,timeout=180)
                self.assertEqual(result.returncode,0,label+'\n'+result.stdout[-2000:]+result.stderr[-8000:])

            for end,start in ((2,0),(4,2)):
                args=dict(base,stop_after=end)
                if start:args['resume_from_checkpoint']=str(root/'stage/checkpoint-2')
                path=root/f'stage{end}.json';write(path,args)
                execute([sys.executable,str(ROOT/'scripts/run_checked_training.py'),
                         '--arguments',str(path),'--receipt',str(root/f'receipt{end}.json'),
                         '--expected-step',str(end),'--start-step',str(start),'--schedule-steps','4',
                         '--',sys.executable,str(ROOT/'train.py'),str(path)],str(end))
                receipt=json.loads((root/f'receipt{end}.json').read_text())
                self.assertEqual((receipt['global_step'],receipt['schedule_steps']),(end,4))
            reference=dict(base,output_dir=str(root/'reference'),stop_after=4)
            write(root/'reference.json',reference)
            execute([sys.executable,str(ROOT/'train.py'),str(root/'reference.json')],'reference')
            from safetensors.torch import load_file
            a=root/'stage/checkpoint-4';b=root/'reference/checkpoint-4'
            self.assertEqual(digest(load_file(str(a/'model.safetensors'))),
                             digest(load_file(str(b/'model.safetensors'))))
            for name in ('optimizer.pt','scheduler.pt','rng_state.pth'):
                self.assertEqual(digest(torch.load(a/name,weights_only=False)),
                                 digest(torch.load(b/name,weights_only=False)),name)


if __name__ == '__main__':
    unittest.main()
