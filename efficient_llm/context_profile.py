"""Actual eager whole-model cached-decode timing for executable static context policies."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import statistics
import torch
import torch.distributed as dist
from efficient_llm.diagnostic import load_model,write
from efficient_llm.diagnostic_data import sha256
from efficient_llm.context_study import policy
from efficient_llm.decode_profile import trial


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','study','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--steps',type=int,default=32);p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args();rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    if a.output.exists() or min(a.steps,a.repeats)<1:raise ValueError('Fresh output and positive sizes required')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)))
    if world>1:dist.init_process_group('nccl')
    try:
        config=json.loads((a.study/'context-config.json').read_text())
        if json.loads((a.study/'evaluation.json').read_text())['status']!='passed':raise ValueError('Quality gate incomplete')
        if rank==0:
            for name,expected in config['model_files'].items():
                if sha256(a.model/name)!=expected:raise ValueError('Changed model')
        if world>1:dist.barrier()
        model=load_model(a.model,torch.device('cuda',torch.cuda.current_device()),torch.bfloat16)
        cases=[(1,2048),(8,2048),(1,8192),(8,8192)]
        assigned=[cases[rank%4]] if world>=4 else cases[rank::world]
        names=['dense','local','fixed_groups','fixed_layers','periodic_layers'];rows=[]
        for batch,length in assigned:
            generator=torch.Generator().manual_seed(42+batch+length)
            ids=torch.randint(0,model.config.vocab_size,(batch,length+4+a.steps),generator=generator).cuda()
            values={name:[] for name in names}
            for repeat in range(-1,a.repeats):
                shift=(rank+repeat+1)%len(names);order=names[shift:]+names[:shift]
                for name in order:
                    with (nullcontext() if name=='dense' else policy(model,name,config)):
                        result=trial(model,ids,length,a.steps)
                    if repeat>=0:values[name].append(result)
            baseline=statistics.median(v['wall_ms'] for v in values['dense'])
            for name in names:
                median=statistics.median(v['wall_ms'] for v in values[name])
                rows.append(dict(rank=rank,batch=batch,context=length,method=name,steps=a.steps,
                    repetitions=values[name],median_ms_per_step=median/a.steps,tokens_per_second=batch*a.steps*1000/median,
                    speedup_vs_dense=baseline/median,gpu=torch.cuda.get_device_name()))
                print('CONTEXT_DECODE',json.dumps(rows[-1]),flush=True)
        gathered=[None]*world
        if world>1:dist.all_gather_object(gathered,rows)
        else:gathered=[rows]
        if rank==0:write(a.output,dict(status='passed',world_size=world,backend='native_sdpa',precision='bf16',
            config_sha256=sha256(a.study/'context-config.json'),results=[r for group in gathered for r in group],
            notes=['Whole-model cached decode, includes projections/MLPs/logits and policy Python overhead.',
                'Fresh StaticCache per repetition; full KV allocation retained; synthetic teacher-forced token IDs.',
                'Policy order rotates per repetition/rank; one full warmup per policy; no collectives during timing.',
                'Eager Hugging Face implementation, no CUDA graphs/compilation; not optimized-serving throughput.',
                'Only static policies execute selective full/local attention; learned routers are not timed.']))
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
