"""Diagnose a BF16 trajectory discrepancy without relaxing the acceptance gate."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from efficient_llm.context_deploy import cached_logits,load_inputs
from efficient_llm.diagnostic import load_probe,write
from efficient_llm import selective_decode


@contextmanager
def routes(routers, replay=None):
    original=routers.layer_choice;seen=[]
    def wrapped(x,index,bank):
        choice=original(x,index,bank)
        if x.shape[1]==1:
            if replay is not None:
                choice=replay[len(seen)]
            seen.append(choice.detach().clone())
        return choice
    routers.layer_choice=wrapped
    try:yield seen
    finally:routers.layer_choice=original


@contextmanager
def fp32_attention():
    original=selective_decode.decode
    def reference(q,k,v,selected,*,length,window,sinks,scale):
        positions=torch.arange(length,device=q.device)
        allowed=selected.repeat_interleave(q.shape[1]//k.shape[1],1)[:,:,None,None] | ((positions<sinks)|(positions>=length-window))[None,None,None,:]
        value=torch.nn.functional.scaled_dot_product_attention(q.float(),k[:,:,:length].float(),v[:,:,:length].float(),allowed,enable_gqa=True,scale=scale)
        return value.transpose(1,2).to(q.dtype)
    selective_decode.decode=reference
    try:yield
    finally:selective_decode.decode=original


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','inputs','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh report required')
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.cuda.set_device(device)
    dist.init_process_group('nccl')
    try:
        torch.set_float32_matmul_precision('highest')
        model,config,routers=load_inputs(a,device,torch.bfloat16)
        _,data=load_probe(a.inputs/'probe');ids=data['evaluation'][rank:rank+1].to(device)
        target=ids[:,2017:2049];scores={};choices={}
        for name in ('dense','dense_kernel','joint_router','joint_kernel'):
            with routes(routers) as seen:scores[name]=cached_logits(model,ids[:,:2048],name,config,routers,2016)
            choices[name]=seen
        with fp32_attention(),routes(routers) as seen:
            scores['joint_math']=cached_logits(model,ids[:,:2048],'joint_kernel',config,routers,2016)
        choices['joint_math']=seen
        with routes(routers,replay=choices['joint_router']) as seen:
            scores['joint_kernel_replay']=cached_logits(model,ids[:,:2048],'joint_kernel',config,routers,2016)
        if len(seen)!=len(choices['joint_router']):raise ValueError('Route replay length mismatch')
        reference=scores['joint_router'];kernel=scores['joint_kernel']
        report=dict(rank=rank,methods={},comparisons={})
        for name,value in scores.items():
            losses=torch.nn.functional.cross_entropy(value.flatten(0,1),target.flatten(),reduction='none')
            report['methods'][name]=dict(nll=losses.cpu().tolist(),mean_nll=losses.mean().item(),predictions=value.argmax(-1).flatten().cpu().tolist())
        for left,right in [('dense','dense_kernel'),('joint_router','joint_kernel'),('joint_router','joint_math'),('joint_kernel','joint_math'),('joint_router','joint_kernel_replay')]:
            item=dict(max_abs_logits=(scores[left]-scores[right]).abs().max().item(),mean_abs_logits=(scores[left]-scores[right]).abs().mean().item(),nll_delta=report['methods'][right]['mean_nll']-report['methods'][left]['mean_nll'])
            if choices.get(left) and choices.get(right):
                lhs=torch.cat([x.flatten() for x in choices[left]]);rhs=torch.cat([x.flatten() for x in choices[right]])
                item.update(route_disagreements=(lhs!=rhs).sum().item(),route_count=lhs.numel())
            report['comparisons'][left+'__'+right]=item
        # Same BF16-rounded weights, FP32 compute: isolates arithmetic/route sensitivity.
        model.float()
        fp_left=cached_logits(model,ids[:,:2048],'joint_router',config,routers,2016)
        fp_right=cached_logits(model,ids[:,:2048],'joint_kernel',config,routers,2016)
        report['fp32_rounded_weights_max_abs']=(fp_left-fp_right).abs().max().item()
        report['fp32_rounded_weights_close']=torch.allclose(fp_left,fp_right,atol=.002,rtol=3e-4)
        write(a.output.with_name(a.output.stem+f'-rank{rank}.json'),report)
        parts=[None]*world;dist.all_gather_object(parts,report)
        if rank==0:
            write(a.output,dict(status='measured',world_size=world,results=parts,
                notes=['Diagnostic only: original BF16 numerical acceptance threshold remains unchanged.',
                       'Math control uses FP32 SDPA on identical BF16 QKV, casts attention output back to BF16.',
                       'Replay forces native route choices but computes its own KV trajectory.']))
            print('NUMERICAL_DIAGNOSIS',json.dumps([dict(rank=r['rank'],comparisons=r['comparisons'],fp32_max_abs=r['fp32_rounded_weights_max_abs']) for r in parts]),flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
