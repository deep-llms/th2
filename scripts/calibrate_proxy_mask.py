"""Optional one-time massive-activation mask from a local trained vanilla checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, TrainingArguments

from deep_kv.model import DeepKV
from deep_kv.packing import preprocess_dataset, isolated_data_collator
from deep_kv.training import DeepKVTrainer
from scripts.check_trained_attention import restore
from train import load_text


@torch.no_grad()
def calibrate(model, batches, token_budget):
    if getattr(model,'proxy_screen',False) or model.arm != 'A':
        raise ValueError('Calibration requires the original vanilla arm A')
    if token_budget <= 0:
        raise ValueError('Calibration token budget must be positive')
    totals = torch.zeros(len(model.backbone.model.layers),model.backbone.config.hidden_size,
                         device=next(model.parameters()).device,dtype=torch.float32)
    count = 0
    handles = []
    for i,layer in enumerate(model.backbone.model.layers):
        def hook(module,args,output,index=i):
            value=output[0] if isinstance(output,tuple) else output
            totals[index].add_(value.float().abs().sum((0,1)))
        handles.append(layer.register_forward_hook(hook))
    previous = model.training
    model.eval()
    try:
        for context in batches:
            if not bool(context.valid.all()):raise ValueError('Calibration expects fully packed contexts')
            model.hidden_states(context)
            count += context.input_ids.numel()
            if count >= token_budget:break
    finally:
        for hook in handles:hook.remove()
        model.train(previous)
    if count < token_budget:raise ValueError('Insufficient calibration tokens')
    means = totals/count
    mask = (means > 20*means.median(-1,keepdim=True).values).any(0)
    if bool(mask.all()):raise ValueError('Calibration excluded every channel')
    return dict(hidden_size=means.shape[1],tokens=count,threshold=20,
                excluded_channels=mask.nonzero().flatten().cpu().tolist(),
                mean_abs_by_block=means.cpu().tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True,help='Old arm-A model.safetensors, local only')
    p.add_argument('--config',required=True)
    p.add_argument('--tokenizer',required=True)
    p.add_argument('--data-dir',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--bf16',action='store_true')
    p.add_argument('--tokens',type=int,default=1048576)
    p.add_argument('--max-documents',type=int,default=20000)
    p.add_argument('--sequence-length',type=int,default=2048)
    p.add_argument('--microbatch',type=int,default=2)
    a=p.parse_args()
    if min(a.tokens,a.max_documents,a.sequence_length,a.microbatch) <= 0:
        raise ValueError('Calibration sizes must be positive')
    if a.output.exists():raise ValueError('Refuse to overwrite channel mask')
    cfg=AutoConfig.from_pretrained(a.config,local_files_only=True);cfg._attn_implementation='sdpa';cfg.use_cache=False
    m=DeepKV.from_scratch(cfg,'A',consumer=1,deep_target=cfg.num_hidden_layers,checkpoint_layers=False)
    restore(m,a.checkpoint);m.to(a.device)
    tokenizer=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    raw=load_text(a.data_dir);raw=raw.select(range(min(len(raw),a.max_documents)))
    args=TrainingArguments(output_dir=str(a.output.parent),report_to=[],use_cpu=a.device=='cpu')
    packed=preprocess_dataset(raw,tokenizer,a.sequence_length,args,num_proc=1,isolate_documents=True)
    loader=DataLoader(packed,batch_size=a.microbatch,collate_fn=isolated_data_collator)
    contexts=(DeepKVTrainer.context({k:v.to(a.device) for k,v in batch.items()}) for batch in loader)
    with torch.autocast(device_type=torch.device(a.device).type,dtype=torch.bfloat16,enabled=a.bf16):
        result=calibrate(m,contexts,a.tokens)
    with a.checkpoint.open('rb') as f:checksum=hashlib.file_digest(f,'sha256').hexdigest()
    result.update(checkpoint=str(a.checkpoint),checkpoint_sha256=checksum,data_dir=a.data_dir,
                  tokenizer=a.tokenizer,packed_fingerprint=packed._fingerprint,bf16=a.bf16)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)


if __name__=='__main__':main()
