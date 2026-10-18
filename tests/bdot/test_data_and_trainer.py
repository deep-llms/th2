import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', HF_HUB_OFFLINE='1')
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, TrainerCallback
from bdot.data import packed_stream
from bdot.trainer import BDOTTrainer
from packed_training import PackedCollator
from train import BaselineTrainingArguments, DataArguments
from training_data import prepare_training_data, require_held_out
from test_model import model

torch.set_num_threads(1)


def tokenizer_assets(path):
    tok=Tokenizer(WordLevel({'[UNK]':0,'[EOS]':1,'a':2,'b':3,'c':4,'d':5},unk_token='[UNK]'))
    tok.pre_tokenizer=Whitespace()
    tok=PreTrainedTokenizerFast(tokenizer_object=tok,unk_token='[UNK]',eos_token='[EOS]',
                               pad_token='[EOS]',model_max_length=32)
    tok.save_pretrained(path)
    return tok


class DataTests(unittest.TestCase):
    def test_manifest_order_cache_and_remainder(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory); tok=tokenizer_assets(p/'tokenizer')
            raw=Dataset.from_dict({'id':[f'd{i}' for i in range(30)],'text':['a b c d '* (i%3+1) for i in range(30)]})
            raw.save_to_disk(p/'raw')
            data=DataArguments(data_dir=str(p/'raw'), data_order='document_stream', dataset_revision='fixture-revision',
                validation_manifest=str(p/'validation.json'),validation_tokens=40,map_batch_size=3,
                data_cache_dir=str(p/'cache'))
            args=BaselineTrainingArguments(output_dir=str(p/'out'),use_cpu=True,report_to='none',seed=42)
            a,ar=prepare_training_data(data,args,tok,p/'tokenizer',8)
            data.map_batch_size=7
            b,br=prepare_training_data(data,args,tok,p/'tokenizer',8)
            self.assertEqual(a.to_dict(),b.to_dict())
            data.preprocessing_num_workers=2
            parallel,_=prepare_training_data(data,args,tok,p/'tokenizer',8)
            self.assertEqual(a.to_dict(),parallel.to_dict())
            data.document_split='validation'
            v,vr=prepare_training_data(data,args,tok,p/'tokenizer',8)
            require_held_out(ar,vr)
            ids=set(json.loads((p/'validation.json').read_text())['document_ids'])
            ordered=raw.shuffle(seed=0)
            expected=[]
            for row in ordered:
                if row['id'] not in ids: expected+=tok(row['text'],add_special_tokens=False)['input_ids']+[1]
            flattened=[x for row in a['input_ids'] for x in row]
            self.assertEqual(flattened,expected[:len(expected)//8*8])
            self.assertGreater(len(v),0)


class RouteTrace(TrainerCallback):
    def __init__(self):self.routes=[];self.lrs=[];self.losses=[]
    def on_substep_end(self,args,state,control,model=None,**kwargs):
        self.routes.append(model.last_routes.clone())
    def on_step_end(self,args,state,control,model=None,optimizer=None,**kwargs):
        self.routes.append(model.last_routes.clone());self.lrs.append(optimizer.param_groups[0]['lr'])


class TrainerTests(unittest.TestCase):
    def make_trainer(self,path,steps=200,cutoff=None,checkpointing=False):
        from train import StopAfterCallback
        m=model(curriculum='A',default_prior=0.,explore_eps_max=.8)
        m.capture_routes=True
        ds=Dataset.from_dict({'input_ids':[[2,3,4,5,6,7],[3,4,5,6,2,8]]*12,
                             'attention_mask':[[1]*6]*24,
                             'labels':[[2,3,4,5,6,7],[3,4,5,6,2,8]]*12,
                             'segments':[[0,0,1,1,1,1],[0,0,0,0,0,0]]*12})
        args=BaselineTrainingArguments(output_dir=str(path),use_cpu=True,report_to='none',
            max_steps=steps,stop_after=cutoff,save_steps=100,logging_steps=100,
            learning_rate=.001,per_device_train_batch_size=1,gradient_accumulation_steps=2,
            remove_unused_columns=False,disable_tqdm=True,seed=5,data_seed=0,
            dataloader_pin_memory=False,train_sampling_strategy='sequential',
            gradient_checkpointing=checkpointing)
        trace=RouteTrace()
        def record_loss(module, inputs, output):
            if module.training and output.loss is not None:
                trace.losses.append(float(output.loss.detach()))
        m.register_forward_hook(record_loss)
        trainer=BDOTTrainer(model=m,args=args,train_dataset=ds,eval_dataset=ds,
            data_collator=PackedCollator(),callbacks=[StopAfterCallback(),trace])
        return trainer,trace

    def test_resume_and_eval(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            full,ft=self.make_trainer(root/'full');full.train()
            first,at=self.make_trainer(root/'split',cutoff=100);first.train()
            resumed,bt=self.make_trainer(root/'split');resumed.train(resume_from_checkpoint=str(root/'split/checkpoint-100'))
            self.assertEqual(len(ft.routes),len(at.routes)+len(bt.routes))
            for a,b in zip(ft.routes,at.routes+bt.routes): torch.testing.assert_close(a,b,atol=0,rtol=0)
            self.assertEqual(ft.lrs,at.lrs+bt.lrs)
            self.assertEqual(ft.losses,at.losses+bt.losses)
            for key,value in full.model.state_dict().items():
                torch.testing.assert_close(value,resumed.model.state_dict()[key],atol=0,rtol=0)
            for opt_id,slot in full.optimizer.state_dict()['state'].items():
                for name,value in slot.items():
                    torch.testing.assert_close(value,resumed.optimizer.state_dict()['state'][opt_id][name],atol=0,rtol=0)
            before=full.model.routing_counter.clone();counts=full.model.load_counts.clone()
            metric=full.evaluate()
            torch.testing.assert_close(before,full.model.routing_counter)
            torch.testing.assert_close(counts,full.model.load_counts)
            self.assertGreater(metric['eval_perplexity'],0)

    def test_checkpointed_trainer(self):
        with tempfile.TemporaryDirectory() as directory:
            a,_=self.make_trainer(Path(directory)/'a',steps=4)
            b,_=self.make_trainer(Path(directory)/'b',steps=4,checkpointing=True)
            a.train();b.train()
            for key,value in a.model.state_dict().items():
                torch.testing.assert_close(value,b.model.state_dict()[key],atol=1e-6,rtol=1e-5)


if __name__=='__main__':unittest.main()
