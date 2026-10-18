"""Offline lm-eval wrapper: native harness task/scoring implementations, custom weights.

One process/device per invocation; independent task/checkpoint invocations may be
scheduled concurrently by the existing runner. No benchmark downloads here.
"""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def make_harness(checkpoint,tokenizer_path=None,device='cpu',batch_size=1,max_length=1024):
    import torch
    from transformers import AutoTokenizer
    from lm_eval.models.huggingface import HFLM
    from train import ModelArguments,build_model
    # Each harness request is one causal text sequence. SDPA supports native
    # padding/cached likelihood calls; packed-only FA4 cannot accept these inputs.
    model=build_model(ModelArguments(model_name_or_path=str(checkpoint),attention_backend='sdpa')).to(device).eval()
    tokenizer=AutoTokenizer.from_pretrained(str(tokenizer_path or checkpoint),local_files_only=True)
    if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
    return HFLM(pretrained=model,tokenizer=tokenizer,backend='causal',device=device,
                batch_size=batch_size,max_length=max_length)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--tokenizer')
    p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--max-length',type=int,default=1024)
    p.add_argument('--tasks',nargs='+',default=['hellaswag','arc_easy','arc_challenge','piqa','winogrande','lambada_openai'])
    a=p.parse_args()
    if int(os.environ.get('WORLD_SIZE','1')) != 1:
        p.error('Use one process per invocation; distribute independent tasks across devices')
    out=Path(a.output)
    if out.exists():p.error('Output already exists')
    from lm_eval import evaluator
    harness=make_harness(a.checkpoint,a.tokenizer,a.device,a.batch_size,a.max_length)
    result=evaluator.simple_evaluate(model=harness,tasks=a.tasks,num_fewshot=0,
        random_seed=0,numpy_random_seed=0,torch_random_seed=0,fewshot_random_seed=0)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as handle:json.dump(result,handle,indent=2,default=str)

if __name__=='__main__':main()
