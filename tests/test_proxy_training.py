"""Real HF Trainer entry points, optimizer-step centering, checkpoint/resume and queues."""
import json
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from datasets import Dataset
from safetensors.torch import load_file, save_file
from transformers import TrainingArguments

import train
from deep_kv import SCREEN_ARMS
from deep_kv.proxy_training import ProxyTrainer, ProxyCallback
from deep_kv.packing import isolated_data_collator
from deep_kv.__main__ import jobs, parse_baselines
from deep_kv.report import report, report_seeds
from tests.test_train import fixture,invoke
from tests.test_proxy_heads import model,cfg,batch,settings


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
                snapshots=[];variance_snapshots=[];inputs_seen=[];calls=[];original=m.update_statistics;forward=m.forward
                def update(sums,squares,counts,initialize=None):
                    calls.append((initialize,counts.clone()))
                    return original(sums,squares,counts,initialize=initialize)
                def observed(*a,**kw):
                    snapshots.append(m.mu.clone());variance_snapshots.append(m.sigma2.clone())
                    inputs_seen.append(a[0].input_ids.clone());out=forward(*a,**kw)
                    if kw.get('compute_auxiliary_losses'):
                        self.assertFalse(out['aux_sum'].requires_grad)
                    return out
                with patch.object(m,'update_statistics',side_effect=update),patch.object(m,'forward',side_effect=observed):
                    trainer.train()
                self.assertEqual([first for first,_ in calls],['mean','variance',None])
                self.assertTrue(torch.equal(calls[0][1],torch.full_like(calls[0][1],8)))
                self.assertTrue(torch.equal(calls[2][1],torch.full_like(calls[2][1],32)))
                for value in snapshots[3:]:torch.testing.assert_close(value,snapshots[2],rtol=0,atol=0)
                self.assertEqual(len(inputs_seen),6)  # Two bootstrap passes plus four real microbatches.
                for value in inputs_seen[1:3]:torch.testing.assert_close(value,inputs_seen[0],rtol=0,atol=0)
                for value in variance_snapshots[3:]:torch.testing.assert_close(value,variance_snapshots[2],rtol=0,atol=0)
                self.assertTrue(torch.equal(calls[1][1],torch.full_like(calls[1][1],8)))
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
                self.assertEqual(args.seed,args.data_seed);self.assertIn(args.seed,(42,1042,2042))
                self.assertFalse(pilot.checkpoint_layers or pilot.checkpoint_lm or pilot.checkpoint_aux)
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

    def test_custom_queues_match_baseline_and_reject_mixed_families(self):
        generated=jobs('deep_kv.b200.json',arms=('A','P1-block'),seeds=(42,))
        for job in generated['jobs']:
            if 'gpus' in job:
                argv=job['argv']
                self.assertEqual(argv[argv.index('--proxy_screen')+1],'true')
        for recipe in ('deep_kv.b200.json','proxy_heads.b200.json'):
            with self.assertRaisesRegex(ValueError,'Cannot mix'):
                jobs(recipe,arms=('B','P1-block'))
        with self.assertRaisesRegex(ValueError,'Cannot mix'):
            jobs('proxy_heads.b200.json',arms=('A','B'))
        with self.assertRaisesRegex(ValueError,'distinct'):
            report_seeds('unused',('A',),(42,42,42))

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

    def test_calibration_saved_proxy_baseline_and_block_capture(self):
        from scripts.calibrate_proxy_mask import calibrate, load_calibration_model
        from deep_kv.model import DeepKV
        ctx=batch()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint=Path(tmp)/'model.safetensors'
            original=model('A')
            # Use changed weights to verify restoration, not matching random initialization.
            with torch.no_grad():original.backbone.model.embed_tokens.weight.mul_(1.2)
            save_file({k:v.clone() for k,v in original.state_dict().items()},checkpoint)
            restored=load_calibration_model(cfg(),checkpoint,proxy_settings=settings())
            for key,value in original.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key],value,rtol=0,atol=0)
            # Independent decoder-forward path gives the same per-block absolute means.
            native=DeepKV.from_scratch(cfg(),'A',consumer=2,deep_target=8,checkpoint_layers=False)
            native.backbone.load_state_dict(original.backbone.state_dict())
            expected=calibrate(native,[ctx],8)
            actual=calibrate(restored,[ctx],8)
            torch.testing.assert_close(torch.tensor(actual['mean_abs_by_block']),
                                       torch.tensor(expected['mean_abs_by_block']),rtol=1e-5,atol=1e-6)
            self.assertEqual(actual['tokens_by_block'],[8]*8)
            self.assertGreater(sum(map(sum,actual['mean_abs_by_block'])),0)
            self.assertTrue(restored.training)
            block=restored.block
            def spiked(index,*args):
                hidden,*rest=block(index,*args)
                if index==3:
                    hidden=hidden.clone();hidden[:,:,7]+=100
                return hidden,*rest
            with patch.object(restored,'block',side_effect=spiked):
                self.assertIn(7,calibrate(restored,[ctx],8)['excluded_channels'])
            with patch.object(restored,'_run_backbone',return_value=None):
                with self.assertRaisesRegex(ValueError,'every block'):
                    calibrate(restored,[ctx],8)
            # Legacy checkpoints remain supported, and unrelated state is never dropped.
            save_file({k:v.clone() for k,v in native.state_dict().items()},checkpoint)
            self.assertIsInstance(load_calibration_model(cfg(),checkpoint),DeepKV)
            bad={k:v.clone() for k,v in original.state_dict().items()};bad['unknown']=torch.zeros(1)
            save_file(bad,checkpoint)
            with self.assertRaisesRegex(RuntimeError,'Unexpected key'):
                load_calibration_model(cfg(),checkpoint,proxy_settings=settings())
        with self.assertRaisesRegex(ValueError,'vanilla arm A'):
            calibrate(model('P1-block'),[ctx],8)

    def test_reused_baseline_queue(self):
        reused=parse_baselines(['42=/existing/seed-42/A'])
        generated=jobs('proxy_heads.b200.json',reuse_baselines=reused)
        training=[j for j in generated['jobs'] if 'gpus' in j]
        self.assertEqual(len(training),23)
        self.assertNotIn('seed-42-arm-A',[j['name'] for j in training])
        self.assertIn('seed-1042-arm-A',[j['name'] for j in training])
        first=generated['jobs'][0]
        self.assertEqual(first['name'],'seed-42-validate-baseline')
        self.assertIn('--expected-step',first['argv']);self.assertIn('2500',first['argv'])
        comparison=next(j for j in generated['jobs'] if j['name']=='seed-42-compare')
        self.assertIn('/existing/seed-42/A',comparison['argv'])
        self.assertIn('42=/existing/seed-42/A',generated['jobs'][-1]['argv'])
        for values in (['42=relative'],['42=/x','42=/y'],['x=/x']):
            with self.assertRaises(ValueError):parse_baselines(values)
        with self.assertRaises(ValueError):jobs('proxy_heads.b200.json',seeds=[1042],reuse_baselines=reused)

    def test_report_target_matching_and_external_baseline(self):
        import copy
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);arms=('A','V1','P1-lambda0','P1-block','P3-block')
            external=root/'old/seed-42/A'
            # Synthetic artifacts test report validation only; no scientific metrics.
            def write_arm(path,arm,seed):
                path.mkdir(parents=True)
                proxy=arm.startswith('P')
                config=dict(pilot=dict(arm=arm,proxy_screen=True,checkpoint_layers=False,
                    checkpoint_lm=False,checkpoint_aux=False,proxy_channel_mask=None),
                    proxy_mask=dict(source='none; no calibration checkpoint supplied',excluded_channels=[]),
                    training=dict(max_steps=10,seed=seed,data_seed=seed,learning_rate=.001),
                    data=dict(eval_rows=2),tokens_per_update=16,train_fingerprint=str(seed),eval_fingerprint='fixed')
                if proxy:
                    config['pilot'].update(proxy_target_version='r7',proxy_variance_floor=.01,
                        proxy_target_clip=10.,proxy_momentum=.99,proxy_loss_form='cosine')
                    config['proxy_target']=dict(target_version='r7',variance_floor=.01,clip=10.,momentum=.99,
                        loss_form='cosine',quantity='mlp_window_sum' if arm.startswith('P1') else 'deep_band_increment',
                        bands=[2] if arm.startswith('P1') else [[2,4]])
                evaluation=dict(eval_rows=2,eval_lm_loss=3.)
                result=dict(arm=arm,global_step=3,schedule_steps=10,input_tokens=48,status='stopped',
                            evaluation=evaluation,training_cost={})
                for name,value in [('train_config',config),('result',result),('eval_results',evaluation),
                                   ('trainer_state',dict(global_step=3,max_steps=10))]:
                    (path/(name+'.json')).write_text(json.dumps(value))
                (path/'model.safetensors').touch()
            for seed in (42,1042):
                for arm in arms:
                    write_arm(external if seed==42 and arm=='A' else root/f'seed-{seed}'/arm,arm,seed)
            before={p.name:p.read_bytes() for p in external.iterdir()}
            summary=report(root/'seed-42',arms,baseline_dir=external,expected_seed=42,expected_step=3)
            self.assertEqual(summary['reused_baseline'],str(external))
            self.assertEqual(set(summary['proxy_targets']),{'P1-lambda0','P1-block','P3-block'})
            report_seeds(root,arms,[42,1042],{42:external})
            self.assertEqual(before,{p.name:p.read_bytes() for p in external.iterdir()})
            for kwargs in ({'expected_seed':1042},{'expected_step':4}):
                with self.assertRaises(ValueError):report(root/'seed-42',arms,baseline_dir=external,**kwargs)
            target=root/'seed-42/P1-block/train_config.json'
            original=json.loads(target.read_text())
            for field,value in [('proxy_channel_mask','/other.json'),('checkpoint_aux',True)]:
                changed=copy.deepcopy(original);changed['pilot'][field]=value
                target.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):report(root/'seed-42',arms,baseline_dir=external)
            changed=copy.deepcopy(original);changed['proxy_target']['clip']=9.
            target.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'target'):report(root/'seed-42',arms,baseline_dir=external)
            target.write_text(json.dumps(original))
            # Even when A is the first arm, masks must match across seeds as well.
            for arm in arms:
                if not arm.startswith('P'):continue
                target=root/f'seed-1042/{arm}/train_config.json';changed=json.loads(target.read_text())
                changed['proxy_target']['clip']=9.;changed['pilot']['proxy_target_clip']=9.;target.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'Seed runs differ in proxy target definition'):
                report_seeds(root,arms,[42,1042],{42:external})


if __name__=='__main__':unittest.main()
