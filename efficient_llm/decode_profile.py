"""Dense native SDPA cached-decode timing; no sparse implementation or speedup claim."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
import torch
import torch.distributed as dist
from efficient_llm.diagnostic import load_model, write
from transformers import StaticCache


class SDPATimer:
    """Time the native core without changing its arguments/output or dispatcher."""
    def __init__(self):
        self.events=[]

    def __enter__(self):
        self.original=torch.nn.functional.scaled_dot_product_attention
        def timed(*args,**kwargs):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record(); result=self.original(*args,**kwargs); end.record()
            self.events.append((start,end))
            return result
        self.wrapper=timed
        torch.nn.functional.scaled_dot_product_attention=timed
        return self

    def __exit__(self,*args):
        torch.nn.functional.scaled_dot_product_attention=self.original

    def milliseconds(self):
        torch.cuda.synchronize()
        return sum(s.elapsed_time(e) for s,e in self.events)


@torch.inference_mode()
def trial(model,ids,context,steps,instrument=False):
    warmup=4
    cache=StaticCache(config=model.config,max_cache_len=context+warmup+steps)
    model(ids[:,:context],past_key_values=cache,use_cache=True,logits_to_keep=1)
    for t in range(context,context+warmup):
        model(ids[:,t:t+1],past_key_values=cache,use_cache=True,logits_to_keep=1)
    torch.cuda.synchronize()
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    timer=SDPATimer() if instrument else None
    if timer: timer.__enter__()
    try:
        wall=time.perf_counter(); start.record()
        for t in range(context+warmup,context+warmup+steps):
            model(ids[:,t:t+1],past_key_values=cache,use_cache=True,logits_to_keep=1)
        end.record(); torch.cuda.synchronize()
        wall_ms=(time.perf_counter()-wall)*1000
    finally:
        if timer: timer.__exit__()
    core=timer.milliseconds() if timer else None
    if timer and len(timer.events)!=model.config.num_hidden_layers*steps:
        raise ValueError('Unexpected SDPA call count; cannot interpret profile')
    return dict(cuda_interval_ms=start.elapsed_time(end),wall_ms=wall_ms,attention_core_ms=core,
                sdpa_calls=len(timer.events) if timer else None)


@torch.inference_mode()
def cache_check(model):
    ids=torch.tensor([[1,4,7,10,13,16,19,22]],device=next(model.parameters()).device)
    dense=model(ids,use_cache=False).logits
    cache=StaticCache(config=model.config,max_cache_len=ids.shape[1]+4)
    parts=[]
    for t in range(ids.shape[1]):
        parts.append(model(ids[:,t:t+1],past_key_values=cache,use_cache=True).logits)
    actual=torch.cat(parts,1)
    torch.testing.assert_close(actual,dense,rtol=3e-4,atol=2e-3)
    return (actual-dense).abs().max().item()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=32);p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args()
    rank,world=int(os.environ.get('RANK',0)),int(os.environ.get('WORLD_SIZE',1))
    if a.output.exists() or min(a.steps,a.repeats)<1: raise ValueError('Require fresh output and positive sizes')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)))
    if world>1: dist.init_process_group('nccl')
    try:
        torch.manual_seed(42);torch.set_float32_matmul_precision('highest')
        model=load_model(a.model,torch.device('cuda',torch.cuda.current_device()),torch.float32)
        cache_error=cache_check(model)
        model.bfloat16()
        cases=[(1,2048),(8,2048),(1,8192),(8,8192)]
        # Eight ranks: two independent device replicates per case. No collectives
        # during timing, so there is no communication in the measured interval.
        assigned=[cases[rank%len(cases)]] if world>=len(cases) else cases[rank::world]
        rows=[]
        for batch,context in assigned:
            g=torch.Generator().manual_seed(42+batch+context)
            ids=torch.randint(0,model.config.vocab_size,(batch,context+4+a.steps),generator=g).cuda()
            trial(model,ids,context,a.steps)  # unreported warmup of full workload
            values=[trial(model,ids,context,a.steps) for _ in range(a.repeats)]
            profile=trial(model,ids,context,a.steps,instrument=True)
            fraction=profile['attention_core_ms']/profile['cuda_interval_ms']
            if not 0<=fraction<=1.01: raise ValueError('Invalid core-time fraction')
            median=statistics.median(v['wall_ms'] for v in values)
            rows.append(dict(rank=rank,gpu=torch.cuda.get_device_name(),batch=batch,context=context,
                steps=a.steps,repetitions=values,median_ms_per_step=median/a.steps,
                tokens_per_second=batch*a.steps*1000/median,instrumented=profile,
                core_fraction_of_instrumented_interval=fraction,
                optimistic_linear_6_of_8_speedup=1/(1-.25*fraction),
                static_cache_fp32_max_abs=cache_error))
            print('DECODE_PROFILE',json.dumps(rows[-1]),flush=True)
        gathered=[None]*world
        if world>1: dist.all_gather_object(gathered,rows)
        else:gathered=[rows]
        if rank==0:
            write(a.output,dict(status='passed',world_size=world,backend='native_sdpa',precision='bf16',
                cache='StaticCache',compiled=False,cuda_graphs=False,synthetic_teacher_forced_tokens=True,
                notes=['Wall latency is measured without timing hooks.',
                       'Attention fraction comes from a separate instrumented pass; hooks add overhead.',
                       'Linear-scaling estimate excludes router/dispatch cost and is not measured sparse speed.',
                       'QKV, output projection, MLP, logits and dense KV storage remain unchanged.',
                       'Results describe this eager HF implementation, not an optimized serving engine.'],
                results=[r for group in gathered for r in group]))
            print('DECODE_PROFILE_COMPLETE',flush=True)
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
