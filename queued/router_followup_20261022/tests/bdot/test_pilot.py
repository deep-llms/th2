"""v1.4 schedule and frozen packed-validation regression tests (CPU/offline)."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from datasets import Dataset
from transformers import get_scheduler

from bdot.routing import RoutingConfig
from bdot.trainer import BDOTTrainer
from bdot.data import select_packed_validation
from packed_training import PackedCollator, PackedTrainer
from train import BaselineTrainingArguments, DataArguments, prepare_validation_data
from training_data import prepare_training_data
from test_data_and_trainer import tokenizer_assets
from test_model import model


class ScheduleTests(unittest.TestCase):
    def test_complete_schedule_and_boundaries(self):
        steps=2_500_000_000//524288
        self.assertEqual(steps,4768)
        warmup=math.ceil(.02*steps)
        self.assertEqual(warmup,96)
        opt=torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))],lr=3e-4)
        scheduler=get_scheduler('cosine_with_min_lr',opt,num_warmup_steps=warmup,
            num_training_steps=steps,scheduler_specific_kwargs={'min_lr_rate':.1})
        lrs=[opt.param_groups[0]['lr']]
        for _ in range(steps):
            opt.step();scheduler.step();lrs.append(opt.param_groups[0]['lr'])
        self.assertEqual(lrs[0],0.)
        self.assertAlmostEqual(lrs[48],1.5e-4)
        self.assertAlmostEqual(lrs[96],3e-4)
        self.assertAlmostEqual(lrs[-1],3e-5)
        self.assertTrue(all(a>=b for a,b in zip(lrs[96:],lrs[97:])))
        a=RoutingConfig(curriculum='A');b=RoutingConfig(curriculum='B')
        self.assertEqual(a.phase(476/steps,True)[0],0)
        self.assertEqual(a.phase(477/steps,True)[0],1)
        self.assertGreater(a.phase(953/steps,True)[2],0)
        self.assertEqual(a.phase(954/steps,True)[2],0)
        self.assertTrue(b.phase(715/steps,True)[1])
        self.assertFalse(b.phase(716/steps,True)[1])
        self.assertGreater(b.phase(2383/steps,True)[2],0)
        self.assertEqual(b.phase(2384/steps,True)[2],0)
        for curriculum in ('A','B'):
            with tempfile.TemporaryDirectory() as tmp:
                args=BaselineTrainingArguments(output_dir=tmp,use_cpu=True,report_to='none',
                    max_steps=steps,warmup_steps=math.ceil(.2*steps),remove_unused_columns=False)
                with self.assertRaisesRegex(ValueError,'warmup'):
                    BDOTTrainer(model=model(curriculum=curriculum),args=args)


class PackedValidationTests(unittest.TestCase):
    def test_overlap_selection_reuse_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);tok=tokenizer_assets(p/'tokenizer')
            shared='a b c d '*8
            Dataset.from_dict({'text':[shared,'c d a b '*8]}).save_to_disk(p/'train')
            Dataset.from_dict({'text':['b a d c '*8,shared,'d c b a '*8]}).save_to_disk(p/'validation')
            args=BaselineTrainingArguments(output_dir=str(p/'out'),use_cpu=True,report_to='none',
                seed=17,data_seed=0)
            data=DataArguments(data_dir=str(p/'train'),eval_data_dir=str(p/'validation'),
                dataset_revision='fixture',validation_manifest=str(p/'manifest.json'),
                map_batch_size=1000,eval_max_sequences=2)
            train,recipe=prepare_training_data(data,args,tok,p/'tokenizer',8)
            original=train.to_dict()
            selected,receipt=prepare_validation_data(data,args,tok,p/'tokenizer',8,'sdpa',recipe)
            manifest=json.loads((p/'manifest.json').read_text())
            self.assertEqual(manifest['overlapping_validation_documents'],[1])
            self.assertGreater(len(manifest['dropped_sequences']),0)
            self.assertTrue(all(1 not in row for row in selected['segments']))
            self.assertEqual(len(selected),2)
            data.eval_max_sequences=None
            with patch('bdot.data.load_text',side_effect=AssertionError('must not rescan text')):
                full,again=prepare_validation_data(data,args,tok,p/'tokenizer',8,'sdpa',recipe)
            self.assertEqual(selected.to_dict(),full.select(range(2)).to_dict())
            self.assertEqual(receipt['manifest'],again['manifest'])
            self.assertEqual(train.to_dict(),original)
            # HF may rewrite tokenizer JSON on save; cache identity is incidental.
            compatible=copy.deepcopy(again['data_recipe'])
            compatible.update(tokenizer_files=[],shuffled_fingerprint='reserialized-tokenizer')
            raw_eval_args=copy.copy(args);raw_eval_args.seed=raw_eval_args.data_seed=0
            eval_data=copy.copy(data);eval_data.data_dir=data.eval_data_dir
            all_rows,_=prepare_training_data(eval_data,raw_eval_args,tok,p/'tokenizer',8)
            self.assertEqual(select_packed_validation(all_rows,compatible,manifest).to_dict(),full.to_dict())
            compatible['block_size']=16
            with self.assertRaisesRegex(ValueError,'validation manifest'):
                select_packed_validation(all_rows,compatible,manifest)
            bad=copy.deepcopy(recipe);bad['raw_fingerprint']='different'
            with self.assertRaisesRegex(ValueError,'manifest differs'):
                prepare_validation_data(data,args,tok,p/'tokenizer',8,'sdpa',bad)


class PipelineTests(unittest.TestCase):
    def test_native_and_fixed_bdot_500_updates(self):
        """T14 CPU fixture: same initial weights, full schedule, data and optimizer."""
        with tempfile.TemporaryDirectory() as tmp:
            routed=model(r=0,routing='fixed',curriculum='none',step_film=False)
            native=routed.native_reference()
            rows=[dict(input_ids=[2,3,4,5,6,7],labels=[2,3,4,5,6,7],
                attention_mask=[1]*6,segments=[0,0,1,1,1,1]),
                dict(input_ids=[3,4,5,6,2,8],labels=[3,4,5,6,2,8],
                attention_mask=[1]*6,segments=[0]*6)]
            dataset=Dataset.from_list(rows*512)
            results=[]
            for name,network,cls in [('native',native,PackedTrainer),('bdot',routed,BDOTTrainer)]:
                losses=[]
                def record(m,inputs,output):
                    if m.training:
                        losses.append(float(output.loss.detach()))
                handle=network.register_forward_hook(record)
                args=BaselineTrainingArguments(output_dir=str(Path(tmp)/name),use_cpu=True,
                    report_to='none',max_steps=500,warmup_steps=10,learning_rate=3e-4,
                    lr_scheduler_type='cosine_with_min_lr',lr_scheduler_kwargs={'min_lr_rate':.1},
                    adam_beta1=.9,adam_beta2=.95,weight_decay=.1,max_grad_norm=1.,
                    per_device_train_batch_size=1,gradient_accumulation_steps=2,
                    per_device_eval_batch_size=2,train_sampling_strategy='sequential',
                    remove_unused_columns=False,save_strategy='no',logging_strategy='no',
                    dataloader_pin_memory=False,disable_tqdm=True,seed=0,data_seed=0)
                t=cls(model=network,args=args,train_dataset=dataset,eval_dataset=Dataset.from_list(rows),
                    data_collator=PackedCollator())
                result=t.train()
                self.assertEqual(len(losses),1000)
                metrics=t.evaluate()
                results.append((sum(losses[800:])/100,metrics['eval_loss']))
                self.assertGreater(result.metrics['train_runtime'],0)
                handle.remove()
            self.assertLess(abs(results[0][0]-results[1][0]),.01)
            self.assertLess(abs(results[0][1]-results[1][1]),.01)


if __name__=='__main__':
    unittest.main()
