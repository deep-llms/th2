"""Populate ordinary HF map caches using train.py's exact preprocessing, on CPU.

No token export or training; this keeps existing GPU burns active during the
long first corpus map. Training calls the same functions and validates its own
receipts normally, whether the caches hit or need rebuilding.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arguments',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Cache receipt already exists')
    from transformers import HfArgumentParser,AutoTokenizer
    from train import ModelArguments,DataArguments,BaselineTrainingArguments,prepare_validation_data
    from training_data import prepare_training_data
    options=json.loads(a.arguments.read_text());options.update(use_cpu=True,bf16=False,tf32=False)
    model,data,args=HfArgumentParser((ModelArguments,DataArguments,BaselineTrainingArguments)).parse_dict(options)
    tokenizer=AutoTokenizer.from_pretrained(model.tokenizer_name,local_files_only=True)
    if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
    start=time.time()
    ds,recipe=prepare_training_data(data,args,tokenizer,model.tokenizer_name,data.block_size,model.attention_backend)
    validation,vr=prepare_validation_data(data,args,tokenizer,model.tokenizer_name,data.block_size,model.attention_backend,recipe)
    steps = args.stop_after if args.stop_after is not None else args.max_steps
    if steps <= 0:
        raise ValueError('Cache verification requires an explicit positive step budget')
    if len(ds)<steps*512:raise ValueError('Insufficient packed data for the authorized budget')
    result=dict(status='passed',seconds=time.time()-start,train_sequences=len(ds),
                validation_sequences=len(validation),data_recipe=recipe,validation_recipe=vr)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print('TRAINING_CACHE_READY '+json.dumps({k:v for k,v in result.items() if not k.endswith('recipe')}),flush=True)


if __name__=='__main__':main()
