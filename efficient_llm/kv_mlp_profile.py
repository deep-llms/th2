"""Whole-model native SDPA prefill/decode cost with real compact cache/weights."""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import time
import torch
import torch.distributed as dist
from transformers import DynamicCache
from efficient_llm.kv_mlp import load_compressed
from efficient_llm.kv_mlp_check import cache_bytes
from efficient_llm.diagnostic import write


@torch.inference_mode()
def trial(model, ids, length, steps=24):
    cache=DynamicCache(config=model.config)
    torch.cuda.synchronize();start=time.perf_counter()
    model(ids[:,:length],past_key_values=cache,use_cache=True,logits_to_keep=1)
    torch.cuda.synchronize();prefill=(time.perf_counter()-start)*1000
    for t in range(length,length+4):model(ids[:,t:t+1],past_key_values=cache,use_cache=True,logits_to_keep=1)
    torch.cuda.synchronize();start=time.perf_counter()
    for t in range(length+4,length+4+steps):
        model(ids[:,t:t+1],past_key_values=cache,use_cache=True,logits_to_keep=1)
    torch.cuda.synchronize();decode=(time.perf_counter()-start)*1000/steps
    return dict(prefill_ms=prefill,decode_ms_per_step=decode,cache_bytes=cache_bytes(cache),
                active_kv_layers=sum(x.is_initialized for x in cache.layers))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    if a.output.exists():raise ValueError('Require fresh profile output')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)))
    if world>1:dist.init_process_group('nccl')
    try:
        rows=[];cases=[(1,2048),(8,2048),(1,8192),(8,8192)]
        assigned=[cases[rank%4]] if world>=4 else cases[rank::world]
        # Rotate arm ordering across devices to reduce time/order bias.
        arms=list('ABCDEF');arms=arms[rank%6:]+arms[:rank%6]
        for arm in arms:
            model=load_compressed(a.runs/arm/'final',torch.device('cuda'),torch.bfloat16).eval()
            for batch,length in assigned:
                ids=torch.randint(model.config.vocab_size,(batch,length+28),generator=torch.Generator().manual_seed(42)).cuda()
                trial(model,ids,length)
                reps=[trial(model,ids,length) for _ in range(3)]
                rows.append(dict(rank=rank,arm=arm,batch=batch,context=length,repetitions=reps,
                    prefill_ms=statistics.median(x['prefill_ms'] for x in reps),
                    decode_ms_per_step=statistics.median(x['decode_ms_per_step'] for x in reps),
                    cache_bytes=reps[0]['cache_bytes'],parameters=sum(p.numel() for p in model.parameters())))
            del model;gc.collect();torch.cuda.empty_cache()
        gathered=[None]*world
        if world>1:dist.all_gather_object(gathered,rows)
        else:gathered[0]=rows
        if rank==0:write(a.output,dict(status='measured',world_size=world,rows=[x for part in gathered for x in part],
            limitations=['Eager native SDPA, BF16 weights, DynamicCache; not a serving benchmark.',
                         'All consumers still perform attention; cache storage is not traffic or latency.',
                         'Synthetic IDs establish cost only; quality measured separately on real text.']))
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
