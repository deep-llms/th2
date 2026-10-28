"""Numerical checks for the *actual* native OpenSci checkpoint, before healing.

These functions do not mark a pilot ready by themselves: aggregate all parents,
metadata, data disjointness, CPU/DDP receipts and whole-workload T8 estimates.
"""
import copy
from pathlib import Path
import torch
from transformers import default_data_collator
from .model import prune,bypass,forward_kl,save_student,reload_student,round_to_saved_precision,KDModel
from .measure import full_fp32,rank_world,residual_profile,score
from .protocol import layers_for


@torch.no_grad()
def parent_sanity(model, small):
    """Cheap weight-dependent forward and config-dependent RoPE formula check."""
    from packed_evaluation import preserve_rng_and_mode
    device = next(model.parameters()).device
    with preserve_rng_and_mode(model), full_fp32():
        model.eval()
        ids = small[0]['input_ids'][None, :64].to(device)
        finite = bool(torch.isfinite(model(ids, use_cache=False).logits).all())
        hd = model.config.head_dim
        expected = 1 / (model.config.rope_theta ** (
            torch.arange(0, hd, 2, device=device, dtype=torch.float32) / hd))
        error = (model.model.rotary_emb.inv_freq - expected).abs().max().item()
    result = dict(finite_logits=finite, T10_formula_max_abs=error, passed=finite and error < 1e-6)
    if not result['passed']:
        raise ValueError('Parent forward/RoPE sanity failed')
    return result


def _numerical_checks(model,small,sets,output):
    """T4/T5/T10 formula/T11/T12, FP32 native model, bounded 64-token probe.

    Full small-set export/score checks are also required (train.heal always does
    those after healing). Test fixtures can call this on a native tiny OpenSci.
    """
    device=next(model.parameters()).device; rank,world=rank_world()
    ids=small[0]['input_ids'][None,:64].to(device)
    checks={}
    with full_fp32(),torch.no_grad():
        reference=model(ids,use_cache=False).logits
        student=copy.deepcopy(model);prune(student,[])
        checks['T4_no_removal']=bool(torch.equal(reference,student(ids,use_cache=False).logits));del student
        n=len(model.model.layers)
        indices=[(0,), (min(11,n-1),), (n-1,)] + [tuple(layers_for(groups)) for groups in sets.values()]
        indices=list(dict.fromkeys(indices))
        checks['removal_patterns']=[list(p) for p in indices]
        diffs=[]
        for removed in indices:
            student=copy.deepcopy(model);prune(student,removed)
            with bypass(model,removed):expected=model(ids,use_cache=False).logits
            diffs.append((student(ids,use_cache=False).logits-expected).abs().max().item())
            del student
        checks['T4_prune_max_abs']=max(diffs)
        theta=model.config.rope_theta; hd=model.config.head_dim
        expected=1/(theta**(torch.arange(0,hd,2,device=device,dtype=torch.float32)/hd))
        checks['T10_formula_max_abs']=(model.model.rotary_emb.inv_freq-expected).abs().max().item()
        with bypass(model,range(n)):
            profile=residual_profile(model,[small[0]],amp=False)
        checks['T11_max_abs']=max(abs(v) for vs in profile.values() for v in vs)
        student=copy.deepcopy(model);prune(student,indices[-1]);round_to_saved_precision(student)
        target=student(ids,use_cache=False).logits
        saved_val=score(student,small)['loss']
        path=Path(output)/'pruned_roundtrip'
        if rank==0:save_student(student,path)
        if world>1:torch.distributed.barrier()
        restored=reload_student(path,device)
        checks['T4_reload_max_abs']=(restored(ids,use_cache=False).logits-target).abs().max().item()
        checks['T4_reload_val_abs']=abs(score(restored,small)['loss']-saved_val)
        del student,restored
        hidden=model.model(ids[:,:8],use_cache=False).last_hidden_state.detach()
    # Exact teacher/student identity. Full native decoder gradient is verified in
    # the CPU/DDP integration tests; here stress the real vocabulary/head sizes.
    weight=model.lm_head.weight
    h=hidden.clone().requires_grad_(True)
    same=forward_kl(h,weight,h.detach(),weight.detach(),4)
    grads=torch.autograd.grad(same,(h,weight))
    checks['T5_loss']=same.item();checks['T5_finite_gradient']=all(bool(torch.isfinite(g).all()) for g in grads)
    del grads,same
    th=hidden.flip(1).contiguous()
    h=hidden.clone().requires_grad_(True)
    full=forward_kl(h,weight,th,weight.detach(),hidden.shape[1],recompute_head=False)
    gfull=torch.autograd.grad(full,(h,weight))
    chunk=forward_kl(h,weight,th,weight.detach(),3)
    gchunk=torch.autograd.grad(chunk,(h,weight))
    checks['T12_loss_abs']=abs(chunk.item()-full.item())
    checks['T12_gradient_max_abs']=max((a-b).abs().max().item() for a,b in zip(gfull,gchunk))
    del gfull,gchunk,full,chunk,h
    # A nonuniform 64-example batch, shared projection, all positions included.
    features=hidden.repeat(64,1,1)
    scale=torch.linspace(.9,1.1,64,device=device)[:,None,None]
    features=features*scale
    transform=torch.eye(features.shape[-1],device=device,requires_grad=True)
    th=features.flip(1).contiguous()
    loss=forward_kl(features@transform,weight,th,weight.detach(),3)
    reference_grads=torch.autograd.grad(loss,(transform,weight))
    accumulated=[torch.zeros_like(g) for g in reference_grads]
    for start in range(0,64,8):
        loss=forward_kl(features[start:start+8]@transform,weight,th[start:start+8],weight.detach(),3)/8
        gradients=torch.autograd.grad(loss,(transform,weight))
        for target,g in zip(accumulated,gradients):target.add_(g)
    checks['T12_accum_max_abs']=max((a-b).abs().max().item() for a,b in zip(reference_grads,accumulated))
    checks['passed']=(checks['T4_no_removal'] and checks['T4_prune_max_abs']<1e-4
        and checks['T4_reload_max_abs']<1e-6 and checks['T4_reload_val_abs']<=1e-6 and checks['T10_formula_max_abs']<1e-6
        and checks['T11_max_abs']<1e-6 and abs(checks['T5_loss'])<1e-6 and checks['T5_finite_gradient']
        and all(checks[k]<=1e-5 for k in ('T12_loss_abs','T12_gradient_max_abs','T12_accum_max_abs')))
    checks['probe_shape']=list(ids.shape)
    return checks


def numerical_checks(model,small,sets,output):
    with full_fp32():
        return _numerical_checks(model,small,sets,output)


def throughput(student,teacher,rows,output,*,microbatch=1,steps=12):
    """T8: discarded fresh model, same full 64x4096 KD workload; no scored arm."""
    import time
    from .train import HealTrainer,RuntimeGuard,make_arguments,audit_order
    rank,world=rank_world(); device=next(student.parameters()).device
    if device.type!='cuda' or world!=8 or rows.length!=4096 or len(rows)!=steps*64:
        raise ValueError('T8 production throughput requires real 8-GPU 64x4096 workload')
    student.requires_grad_(True)
    args=make_arguments(output,steps,1e-4,microbatch,world,cuda=True)
    class Timer(RuntimeGuard):
        def __init__(self):super().__init__(3600);self.times=[]
        def on_step_begin(self,*args,**kwargs):
            super().on_step_begin(*args,**kwargs)
            torch.cuda.synchronize();self.step_started=time.perf_counter()
        def on_step_end(self,*args,**kwargs):
            torch.cuda.synchronize();self.times.append(time.perf_counter()-self.step_started)
    timer=Timer()
    trainer=HealTrainer(model=KDModel(student),teacher=teacher,args=args,
        train_dataset=rows,data_collator=default_data_collator,callbacks=[timer])
    trainer.train()
    traces=[None]*world;torch.distributed.all_gather_object(traces,trainer.seen)
    coverage=audit_order(traces,rows.order.tolist(),world,microbatch)
    times=torch.tensor(timer.times,device=device);torch.distributed.all_reduce(times,op=torch.distributed.ReduceOp.MAX)
    # Exclude only the first two startup steps, report all durations as well.
    seconds=times[2:].mean().item()
    return dict(status='passed',step_seconds=seconds,tokens_per_second=262144/seconds,
        all_step_seconds=times.tolist(),gpu=torch.cuda.get_device_name(),world_size=world,
        max_memory_bytes=torch.cuda.max_memory_allocated(),coverage=coverage,
        note='Not a healing arm. Excludes evaluations/load/save; full-workload estimate still required.')
