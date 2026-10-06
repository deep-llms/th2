"""Disposable CUDA checks for P7 gradient isolation and the batched KL optimization."""
import argparse
import gc
import statistics

import torch
from transformers import AutoConfig

from deep_kv import MEMORY_ARMS
from deep_kv.packing import isolated_data_collator
from deep_kv.proxy import ProxyModel
from deep_kv.proxy_memory import MemoryPlan, relational_losses
from deep_kv.proxy_training import ProxyTrainer
from scripts.check_fa4_proxy import read, write
from tests.test_proxy_memory import loop_relational_reference


def relational_check():
    """Same FP32 objective/derivatives at production shape, including padded queries."""
    torch.manual_seed(71)
    documents=torch.arange(2048,device='cuda')[None].expand(16,-1).clone()//509
    documents[0]=torch.arange(2048,device='cuda')  # No eligible queries.
    documents[1]=torch.arange(2048,device='cuda')//2  # Mixed fragments.
    plan=MemoryPlan(documents,step=125,sequence_indices=list(range(16)))
    target=torch.randn(16,2048,1024,device='cuda')
    value=torch.randn_like(target)
    results=[]
    for quantize in (False,True):
        # Check FP32 derivatives also at BF16-representable model outputs.
        pred=(value.bfloat16().float() if quantize else value).requires_grad_()
        captures=[]
        for function in (loop_relational_reference,relational_losses):
            sums,counts=function(pred,target,plan)
            loss=sums.sum()/counts.sum()
            grad,=torch.autograd.grad(loss,pred)
            captures.append((loss.detach(),counts,grad))
        a,b=captures
        torch.testing.assert_close(a[0],b[0],rtol=2e-5,atol=2e-6)
        torch.testing.assert_close(a[1],b[1],rtol=0,atol=0)
        torch.testing.assert_close(a[2],b[2],rtol=2e-4,atol=2e-8)
        assert b[2][0].count_nonzero()==0
        results.append(dict(bf16_inputs=quantize,loss_difference=float((a[0]-b[0]).abs()),
            gradient_max_abs=float((a[2]-b[2]).abs().max()),
            gradient_relative_l2=float((a[2]-b[2]).norm()/a[2].norm())))
    times={'loop':[],'batched':[]}
    pred=value.requires_grad_()
    for repeat in range(11):
        order=[('loop',loop_relational_reference),('batched',relational_losses)]
        if repeat%2:order.reverse()
        for name,function in order:
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            sums,counts=function(pred,target,plan)
            torch.autograd.grad(sums.sum()/counts.sum(),pred)
            end.record();end.synchronize()
            if repeat>=2:times[name].append(start.elapsed_time(end))
    return dict(status='passed',shape=list(value.shape),comparisons=results,
        median_forward_backward_ms={k:statistics.median(v) for k,v in times.items()},
        note='One layer, fixed query plan; end-to-end Trainer timing reported separately.')


def routing_check(args):
    recipe=read(args.recipe)
    cfg=AutoConfig.from_pretrained(recipe['config_name'],local_files_only=True)
    cfg._attn_implementation='sdpa';cfg.use_cache=False
    rows=read(args.rows)[:2]
    assert len(rows)==2 and all(len(r['input_ids'])==2048 for r in rows)
    ctx=ProxyTrainer.context({k:v.cuda() for k,v in isolated_data_collator(rows).items()})
    cases=[]
    for backend in ('sdpa','fa4'):
        for arm in MEMORY_ARMS:
            print('ROUTING',arm,backend,flush=True)
            model=ProxyModel.from_scratch(cfg,arm,attention_backend=backend,seed=42,
                checkpoint_layers=False,checkpoint_lm=False,checkpoint_aux=False,lm_chunk=128).cuda().train()
            with torch.no_grad():
                for phase in ('mean','variance'):
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        out=model(ctx,compute_auxiliary_losses=False,auxiliary_grad=False,
                            collect_target_statistics=True,statistics_mode=phase)
                    model.update_statistics(out['center_sums'],out['center_squares'],out['center_counts'],initialize=phase)
            buffers={n:v.clone() for n,v in model.named_buffers()}
            counts={}
            for component in (('lm','cosine','relational') if model.relational_proxy else ('lm','cosine')):
                model.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    out=model(ctx,relational_step=125)
                    losses={'lm':out['lm_sum']/out['lm_count'],'cosine':out['aux_sum']/out['aux_count']}
                    if model.relational_proxy:losses['relational']=out['rel_sum']/out['rel_count']
                    loss=losses[component]
                loss.backward()
                active_count=0
                for name,p in model.named_parameters():
                    if p.grad is not None:assert torch.isfinite(p.grad).all(),name
                    active=p.grad is not None and bool(p.grad.count_nonzero())
                    estimator=name.startswith('heads.') and any(part in name for part in ('.conv','.w_in.','.w_out.','.w1.','.w2.'))
                    # All native/proxy projection parameters receive LM gradients;
                    # only estimator parameters receive either auxiliary gradient.
                    assert active==(not estimator if component=='lm' else estimator),(arm,backend,component,name)
                    active_count+=active
                assert torch.isfinite(loss)
                counts[component]=dict(loss=float(loss.detach()),active_parameters=active_count)
                for n,v in model.named_buffers():assert torch.equal(v,buffers[n]),n
                del out,loss
            cases.append(dict(arm=arm,backend=backend,loss_routes=counts,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
            del model,buffers
            gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    return cases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('recipe','rows','output'):p.add_argument('--'+key,required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    component=relational_check()
    print('RELATIONAL',component,flush=True)
    cases=routing_check(args)
    write(args.output,dict(status='passed',relational=component,routing=cases))


if __name__=='__main__':main()
