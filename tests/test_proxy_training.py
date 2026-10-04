"""Real HF Trainer entry points, optimizer-step centering, checkpoint/resume and queues."""
import json
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from datasets import Dataset
from safetensors.torch import load_file
from transformers import TrainingArguments

import train
from deep_kv import SCREEN_ARMS
from deep_kv.proxy_training import ProxyTrainer, ProxyCallback
from deep_kv.packing import isolated_data_collator
from deep_kv.__main__ import jobs
from deep_kv.report import report, report_seeds
from tests.test_train import fixture,invoke
from tests.test_proxy_heads import model,cfg,batch


def proxy_fixture(root,world=1):
    config=fixture(root,world=world)
    cfg().save_pretrained(root/'model')
    return {**config,'deep_target':8,'proxy_screen':True,'proxy_groups':1,'proxy_width':8,
            'proxy_features':9,'proxy_chunk_size':2,'proxy_warmup_steps':1,'logging_steps':1}


class ProxyTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_real_entry_all_screen_arms_resume_buffers_and_reliance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=proxy_fixture(root)
            for arm in SCREEN_ARMS:
                args={**config,'arm':arm,'output_dir':str(root/arm)}
                invoke(root,{**args,'stop_after':2})
                saved=load_file(root/arm/'checkpoint-2/model.safetensors')
                self.assertTrue(saved['mu_initialized'])
                if arm.startswith('P'):self.assertGreater(saved['mu'].abs().sum(),0)
                invoke(root,args)
                invoke(root,{**args,'output_dir':str(root/('full-'+arm))})
                resumed=load_file(root/arm/'model.safetensors');full=load_file(root/('full-'+arm)/'model.safetensors')
                for name in full:torch.testing.assert_close(resumed[name],full[name],rtol=0,atol=0,msg=arm+'/'+name)
                result=json.loads((root/arm/'result.json').read_text())
                self.assertEqual(result['global_step'],3)
                self.assertTrue(result['proxy']['mean_initialized'])
                history=json.loads((root/arm/'trainer_state.json').read_text())['log_history']
                logs=[row for row in history if 'seconds_per_update' in row]
                self.assertTrue(logs)
                self.assertTrue(all(row['seconds_per_update']>0 for row in logs))
                if arm.startswith('P'):
                    self.assertIn('eval_reliance_loss_increase',result['evaluation'])
                    self.assertTrue(any('step_proxy_layer_2_cosine_0' in row for row in logs))
                    if arm.endswith('lambda0'):self.assertTrue(all(row['proxy_lambda']==0 for row in logs))
            summary=report(root,SCREEN_ARMS)
            self.assertIn('P1-block-P1-lambda0',summary['nll_differences'])
            self.assertIn('P3-block-V3',summary['nll_differences'])
            # Synthetic copies exercise report recipe checks; these are not new seed runs.
            for seed in (42,43,44):
                for arm in SCREEN_ARMS:
                    dest=root/f'seed-{seed}'/arm
                    dest.mkdir(parents=True)
                    for name in ('train_config.json','result.json','trainer_state.json','eval_results.json','model.safetensors'):
                        shutil.copyfile(root/arm/name,dest/name)
                    path=dest/'train_config.json';recipe=json.loads(path.read_text())
                    recipe['training'].update(seed=seed,data_seed=seed)
                    recipe['train_fingerprint']=f'synthetic-seed-{seed}'
                    path.write_text(json.dumps(recipe))
            combined=report_seeds(root,SCREEN_ARMS,(42,43,44))
            self.assertEqual(combined['lm_loss']['A']['stdev'],0)
            # A matching change across arms is legal within a seed, but not across seeds.
            for arm in SCREEN_ARMS:
                path=root/'seed-44'/arm/'train_config.json';recipe=json.loads(path.read_text())
                recipe['training']['learning_rate']=1.
                path.write_text(json.dumps(recipe))
            with self.assertRaisesRegex(ValueError,'Seed runs differ'):
                report_seeds(root,SCREEN_ARMS,(42,43,44))

    def test_mean_initialization_accumulation_and_zero_lambda_diagnostics(self):
        contexts=batch()
        rows=[dict(input_ids=contexts.input_ids[0].tolist(),labels=contexts.labels[0].tolist(),
                   attention_mask=[1]*8,segments=contexts.segments[0].tolist())]*4
        for arm in ('P1-lambda0','P3-lambda0'):
            with tempfile.TemporaryDirectory() as tmp:
                m=model(arm,checkpoint=True)
                args=TrainingArguments(output_dir=tmp,use_cpu=True,report_to=[],max_steps=1,learning_rate=0.,
                    per_device_train_batch_size=1,gradient_accumulation_steps=4,logging_steps=1,
                    save_strategy='no',remove_unused_columns=False,disable_tqdm=True)
                callback=ProxyCallback(1,32)
                trainer=ProxyTrainer(model=m,args=args,train_dataset=Dataset.from_list(rows),
                    data_collator=isolated_data_collator,callbacks=[callback]);callback.trainer=trainer
                snapshots=[];calls=[];original=m.update_mu;forward=m.forward
                def update(sums,counts,initialize=False):
                    calls.append((initialize,counts.clone()))
                    original(sums,counts,initialize=initialize)
                def observed(*a,**kw):
                    snapshots.append(m.mu.clone());out=forward(*a,**kw)
                    if kw.get('compute_auxiliary_losses'):
                        self.assertFalse(out['aux_sum'].requires_grad)
                    return out
                with patch.object(m,'update_mu',side_effect=update),patch.object(m,'forward',side_effect=observed):
                    trainer.train()
                self.assertEqual([first for first,_ in calls],[True,False])
                self.assertTrue(torch.equal(calls[0][1],torch.full_like(calls[0][1],8)))
                self.assertTrue(torch.equal(calls[1][1],torch.full_like(calls[1][1],32)))
                for value in snapshots[2:]:torch.testing.assert_close(value,snapshots[1],rtol=0,atol=0)
                self.assertIsNone(trainer.center_totals)

    def test_queue_three_seeds_all_arms_and_native_parser(self):
        from run_experiments import load_jobs
        with tempfile.TemporaryDirectory() as tmp:
            generated=jobs('proxy_heads.b200.json')
            path=Path(tmp)/'jobs.json';path.write_text(json.dumps(generated));load_jobs(path)
            self.assertEqual(len(generated['jobs']),28)
            training=[job for job in generated['jobs'] if 'gpus' in job]
            self.assertEqual(len(training),24)
            parser=train.HfArgumentParser((train.ModelArguments,train.DataArguments,train.PilotArguments,TrainingArguments))
            for job in training:
                self.assertEqual(job['gpus'],list(range(8)))
                argv=job['argv'];start=next(i for i,a in enumerate(argv) if a.endswith('/train.py'))+1
                model_args,data,pilot,args=parser.parse_args_into_dataclasses(argv[start:]+['--use_cpu','true','--bf16','false','--report_to','none'])
                self.assertTrue(pilot.proxy_screen);self.assertTrue(data.isolate_documents)
                self.assertEqual(args.seed,args.data_seed);self.assertIn(args.seed,(42,43,44))
                self.assertEqual(args.max_steps,28600);self.assertEqual(args.warmup_steps,1430)
                self.assertEqual(pilot.stop_after,2500)
                self.assertIn(f'seed-{args.seed}',args.output_dir)

    def test_zero_lambda_skips_cosines_between_logs_but_collects_means(self):
        for arm in ('P1-lambda0','P3-lambda0'):
            with tempfile.TemporaryDirectory() as tmp:
                m=model(arm)
                trainer=ProxyTrainer(model=m,args=TrainingArguments(output_dir=tmp,use_cpu=True,report_to=[],logging_steps=2))
                trainer.is_in_train=True
                # Normally populated from TrainingArguments by Trainer.train().
                trainer.state.logging_steps=2
                ctx=batch();inputs=dict(input_ids=ctx.input_ids,labels=ctx.labels,
                    attention_mask=ctx.valid,position_ids=ctx.position_ids,segments=ctx.segments)
                for step,expected in ((0,False),(1,True)):
                    trainer.state.global_step=step
                    with patch.object(m,'layer_cosines',wraps=m.layer_cosines) as cosine:
                        loss,out=trainer.compute_loss(m,inputs,return_outputs=True)
                    self.assertEqual(bool(cosine.call_count),expected)
                    self.assertFalse(out['aux_sum'].requires_grad if expected else False)
                    self.assertTrue((out['center_counts']>0).all())
                    self.assertTrue(loss.requires_grad)

    def test_calibration_mask_union(self):
        from scripts.calibrate_proxy_mask import calibrate
        ctx=batch()
        # Calibrator targets the old vanilla model path, with observable block outputs.
        m=__import__('deep_kv.model',fromlist=['DeepKV']).DeepKV.from_scratch(cfg(),'A',consumer=2,deep_target=8,checkpoint_layers=False)
        original=m.backbone.model.layers[3].forward
        def spiked(*args,**kwargs):
            output=original(*args,**kwargs).clone();output[:,:,7]+=100
            return output
        with patch.object(m.backbone.model.layers[3],'forward',side_effect=spiked):
            result=calibrate(m,[ctx],8)
        self.assertIn(7,result['excluded_channels']);self.assertEqual(result['tokens'],8)


if __name__=='__main__':unittest.main()
