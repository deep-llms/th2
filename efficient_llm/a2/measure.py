"""Token-weighted CE, residual-stream diagnostics and controlled perturbations."""
from contextlib import contextmanager
import statistics
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from packed_evaluation import preserve_rng_and_mode
from .model import autocast, forward_kl, bypass


def rank_world():
    return (dist.get_rank(),dist.get_world_size()) if dist.is_initialized() else (0,1)


def reduce_sum(tensor):
    if dist.is_initialized(): dist.all_reduce(tensor)
    return tensor


def document_mask(starts, dtype):
    """starts marks real document boundaries, independent of literal EOS in text."""
    starts=starts.clone(); starts[:,0]=True
    groups=starts.long().cumsum(-1)
    t=starts.shape[1]; ix=torch.arange(t,device=starts.device)
    allowed=(groups[:,:,None]==groups[:,None,:]) & (ix[:,None]>=ix[None,:])
    mask=torch.zeros(allowed.shape,device=starts.device,dtype=dtype).masked_fill(~allowed,torch.finfo(dtype).min)
    last=torch.where(starts,ix,0).cummax(-1).values
    return mask[:,None], ix-last


@torch.no_grad()
def score(model, rows, *, amp=True, batch=1, chunk=128, isolate=False, distributed=True, teacher=None):
    device=next(model.parameters()).device
    rank,world=rank_world() if distributed else (0,1)
    n=len(rows)
    totals=torch.zeros(n,device=device,dtype=torch.float64)
    counts=torch.zeros(n,device=device,dtype=torch.int64)
    bucket=torch.zeros(3,2,device=device,dtype=torch.float64)
    kd=torch.zeros(2,device=device,dtype=torch.float64)
    indices=list(range(rank,n,world))
    with preserve_rng_and_mode(model):
        model.eval()
        for start in range(0,len(indices),batch):
            selection=indices[start:start+batch]
            ids=torch.stack([rows[i]['input_ids'] for i in selection]).to(device)
            kwargs={}
            if isolate:
                mask,pos=document_mask(torch.stack([rows.boundaries(i) for i in selection]).to(device),next(model.parameters()).dtype)
                kwargs=dict(attention_mask=mask,position_ids=pos)
            with autocast(device,amp):
                hidden=model.model(ids,use_cache=False,return_dict=True,**kwargs).last_hidden_state
                if teacher is not None:
                    th=teacher.model(ids,use_cache=False,return_dict=True).last_hidden_state
                    kl=forward_kl(hidden,model.lm_head.weight,th,teacher.lm_head.weight,chunk,recompute_head=False)
                    kd[0] += kl.double()*ids.numel(); kd[1] += ids.numel()
                for offset in range(0,ids.shape[1]-1,chunk):
                    end=min(offset+chunk,ids.shape[1]-1)
                    logits=model.lm_head(hidden[:,offset:end]).float()
                    ce=F.cross_entropy(logits.flatten(0,1),ids[:,offset+1:end+1].reshape(-1),reduction='none').view(len(selection),-1)
                    totals[selection] += ce.double().sum(-1); counts[selection] += ce.shape[1]
                    # Bucket by TARGET position: there is no prediction for position 0.
                    positions=torch.arange(offset+1,end+1,device=device)
                    for b,(low,high) in enumerate(((0,512),(512,2048),(2048,4096))):
                        take=(positions>=low)&(positions<high)
                        bucket[b,0] += ce[:,take].double().sum(); bucket[b,1] += take.sum()*len(selection)
    if distributed:
        for value in (totals,counts,bucket,kd): reduce_sum(value)
    expected=rows[0]['input_ids'].numel()-1
    if not bool((counts==expected).all()): raise ValueError('Missing/duplicated validation coverage')
    if not torch.isfinite(totals).all() or not torch.isfinite(kd).all(): raise FloatingPointError('Nonfinite CE')
    return dict(loss=(totals.sum()/counts.sum()).item(),tokens=int(counts.sum()),sequences=n,
        per_sequence_nll=(totals/counts).cpu().tolist(),
        bucket_losses=[(v/c).item() if c else None for v,c in bucket],bucket_axis='target_position',
        final_kl=(kd[0]/kd[1]).item() if kd[1] else None,
        precision='bf16_autocast' if amp and device.type=='cuda' else 'fp32',isolate=isolate)


@torch.no_grad()
def residual_profile(model,rows,*,amp=True):
    layers=model.model.layers; device=next(model.parameters()).device
    n=len(layers); rank,world=rank_world()
    values=torch.zeros(n,3,device=device,dtype=torch.float64)
    groups=torch.zeros(n//2,2,device=device,dtype=torch.float64)
    inputs={}; group_inputs={}; handles=[]
    def pre(i):
        def hook(module,args,kwargs):
            x=args[0] if args else kwargs['hidden_states']
            inputs[i]=x.detach()
            if i%2==0: group_inputs[i//2]=x.detach()
        return hook
    def post(i):
        def hook(module,args,output):
            y=(output[0] if isinstance(output,tuple) else output).detach().float()
            x=inputs.pop(i).float(); count=x.shape[0]*x.shape[1]
            values[i,0] += (1-F.cosine_similarity(x,y,dim=-1,eps=1e-8)).double().sum()
            values[i,1] += ((y-x).norm(dim=-1)/x.norm(dim=-1).clamp_min(1e-8)).double().sum()
            values[i,2] += count
            if i%2:
                gx=group_inputs.pop(i//2).float()
                groups[i//2,0] += (1-F.cosine_similarity(gx,y,dim=-1,eps=1e-8)).double().sum()
                groups[i//2,1] += count
        return hook
    try:
        for i,layer in enumerate(layers):
            handles.extend((layer.register_forward_pre_hook(pre(i),with_kwargs=True),layer.register_forward_hook(post(i))))
        with preserve_rng_and_mode(model):
            model.eval()
            for i in range(rank,len(rows),world):
                with autocast(device,amp): model.model(rows[i]['input_ids'][None].to(device),use_cache=False)
        reduce_sum(values); reduce_sum(groups)
        if not bool((values[:,2]>0).all()) or not bool((groups[:,1]>0).all()): raise ValueError('Empty hook measurements')
        return dict(layer_bi=(values[:,0]/values[:,2]).cpu().tolist(),
                    relative_update=(values[:,1]/values[:,2]).cpu().tolist(),
                    group_bi=(groups[:,0]/groups[:,1]).cpu().tolist())
    finally:
        for h in handles:h.remove()


@contextmanager
def full_fp32():
    matmul, cudnn=torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    try: yield
    finally:
        torch.backends.cuda.matmul.allow_tf32=matmul; torch.backends.cudnn.allow_tf32=cudnn


@torch.no_grad()
def gaussian(model,rows):
    matrices=[module.weight for layer in model.model.layers for module in layer.modules() if isinstance(module,nn.Linear)]
    if any(w.dtype != torch.float32 for w in matrices): raise ValueError('Gaussian requires FP32 weights')
    # Exact restore, not add/subtract noise (which would accumulate rounding).
    originals=[w.detach().cpu().clone() for w in matrices]
    results={}
    try:
        with full_fp32():
            base=score(model,rows,amp=False)['loss']
            for sigma in (.005,.01,.02):
                deltas=[]
                for seed in (0,1,2):
                    generator=torch.Generator(device=matrices[0].device).manual_seed(seed)
                    for weight,original in zip(matrices,originals,strict=True):
                        w=original.to(weight.device)
                        noise=torch.randn(weight.shape,device=weight.device,dtype=torch.float32,generator=generator)
                        weight.copy_(w+sigma*w.square().mean().sqrt()*noise)
                    deltas.append(score(model,rows,amp=False)['loss']-base)
                results[str(sigma)]=dict(deltas=deltas,mean=statistics.mean(deltas),std=statistics.pstdev(deltas))
    finally:
        for weight,original in zip(matrices,originals,strict=True): weight.copy_(original)
    return dict(base_loss=base, sigmas=results)


def survey(model,small,*,include_gaussian=True):
    with full_fp32(): fp32=score(model,small,amp=False)
    normal=score(model,small)
    profile=residual_profile(model,small)
    bypasses=[]
    for i in range(len(model.model.layers)):
        with bypass(model,[i]): bypasses.append(score(model,small)['loss']-normal['loss'])
    return dict(small=normal,small_fp32=fp32,precision_gap=abs(normal['loss']-fp32['loss']),
        bypass_delta=bypasses,**profile,document_mask=score(model,small,isolate=True),
        gaussian=gaussian(model,small) if include_gaussian else None)
