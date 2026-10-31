"""Registered A2 M measurements; no fitting, quantization or training."""
from contextlib import contextmanager
import numpy as np
import torch
from torch.nn import functional as F
from .model import autocast


def weight_statistics(w, quantized):
    """FP64 moments; scored parent and existing saved RTN weights only."""
    x=w.double(); centered=x-x.mean(); variance=centered.square().mean()
    groups=x.reshape(-1,128); rms=groups.square().mean(-1).sqrt()
    ratio=(groups.max(-1).values-groups.min(-1).values)/rms.clamp_min(1e-300)
    denom=x.square().sum()
    return dict(weights=x.numel(), excess_kurtosis=float(centered.pow(4).mean()/variance.square()-3) if variance else None,
        group_range_over_rms=float(ratio.mean()),zero_rms_groups=int((rms==0).sum()),
        **{name:float((q.double()-x).square().sum()/denom) if denom else None for name,q in quantized.items()})


def noise_seed(seed, matrix_index):
    # Injective for the registered seed/index domain; independent CUDA generators.
    return 202610310000 + 1000*seed + matrix_index


def perturb(w, kind, seed, matrix_index, sigma=None):
    generator=torch.Generator(device=w.device).manual_seed(noise_seed(seed,matrix_index))
    if kind=='iso':
        noise=torch.randn(w.shape,device=w.device,dtype=torch.float32,generator=generator)*(sigma*w.square().mean().sqrt())
        check=None
    else:
        from .quant import grid
        scale,zero,constant,lo=grid(w.reshape(-1,128),3)
        scale=torch.where(constant,torch.zeros_like(scale),scale)
        noise=(torch.rand(w.reshape(-1,128).shape,device=w.device,generator=generator)-.5)*scale
        expected=scale.double().square().mean()/12
        actual=noise.double().var(unbiased=False)
        excess=(noise.abs()-scale/2).max()
        ratio=float(actual/expected) if expected else None
        check=dict(max_bound_excess=float(excess),variance=float(actual),expected_variance=float(expected),
            variance_ratio=ratio,passed=bool(excess<=0) and (abs(ratio-1)<=.05 if ratio is not None else bool((noise==0).all())))
        noise=noise.reshape_as(w)
    return (w+noise).bfloat16().float(),check


def pair_statistics(x,y):
    x=x.float();y=y.float()
    # Exact identity is mathematically BI=0; avoid cosine's roundoff at x==y.
    cos=F.cosine_similarity(x,y,dim=-1,eps=1e-8)
    bi=torch.where((x==y).all(-1),torch.zeros_like(cos),1-cos)
    update=(y-x).norm(dim=-1)/x.norm(dim=-1).clamp_min(1e-8)
    return torch.stack((bi.double().sum(),update.double().sum(),bi.new_tensor(bi.numel(),dtype=torch.float64)))


@contextmanager
def profile_hooks(model):
    """One traversal, all positions including last; S1 h16 -> h22."""
    device=next(model.parameters()).device
    totals=torch.zeros(len(model.model.layers),3,dtype=torch.float64,device=device)
    block=torch.zeros(3,dtype=torch.float64,device=device);inputs={};handles=[]
    def before(i):
        def hook(module,args,kwargs):
            x=args[0] if args else kwargs['hidden_states'];inputs[i]=x.detach()
            if i==16:inputs['S1']=x.detach()
        return hook
    def after(i):
        def hook(module,args,out):
            y=out[0] if isinstance(out,tuple) else out
            totals[i].add_(pair_statistics(inputs.pop(i),y))
            if i==21:block.add_(pair_statistics(inputs.pop('S1'),y))
        return hook
    try:
        for i,l in enumerate(model.model.layers):
            handles.extend([l.register_forward_pre_hook(before(i),with_kwargs=True),l.register_forward_hook(after(i))])
        yield totals,block
    finally:
        for h in handles:h.remove()


@torch.no_grad()
def token_score(model, rows, output):
    """Same batch=1, head chunks=128 and CE as A2 Q; store exact float32 CE."""
    device=next(model.parameters()).device;model.eval()
    arr=np.lib.format.open_memmap(output,mode='w+',dtype=np.float32,shape=(len(rows),rows[0]['input_ids'].numel()-1))
    independent=[]
    for i in range(len(rows)):
        total=torch.zeros((),device=device,dtype=torch.float64)
        ids=rows[i]['input_ids'][None].to(device)
        with autocast(device):
            hidden=model.model(ids,use_cache=False,return_dict=True).last_hidden_state
            for start in range(0,ids.shape[1]-1,128):
                end=min(start+128,ids.shape[1]-1)
                logits=model.lm_head(hidden[:,start:end]).float()
                ce=F.cross_entropy(logits.flatten(0,1),ids[:,start+1:end+1].reshape(-1),reduction='none')
                total+=ce.double().sum()
                arr[i,start:end]=ce.cpu().numpy()
        independent.append(float(total)/(ids.shape[1]-1))
        if i%250==0:print('MECH_TOKEN_PROGRESS',i,len(rows),flush=True)
    arr.flush();means=np.asarray(arr,dtype=np.float64).mean(-1)
    return dict(loss=float(means.mean()),per_sequence_nll=means.tolist(),sequences=len(rows),tokens=arr.size,
        storage_replay_max_abs=float(np.max(np.abs(means-independent))))


def token_buckets(values, targets):
    names=('P300','P1T','S72','S188','pre-P300','pre-P1T','student-P300','student-P1T','student-X72')
    if set(values)!=set(names):raise ValueError('Incomplete token model coverage')
    if any(v.shape!=targets.shape for v in values.values()):raise ValueError('Unpaired token shapes')
    target=targets.reshape(-1);count=np.bincount(target);out={}
    for kind,key in [('frequency',count[target]),('difficulty',values['S188'].reshape(-1))]:
        order=np.argsort(key,kind='stable') # ties resolved by sequence then target position
        buckets=[]
        for index,ix in enumerate(np.array_split(order,10)):
            means={n:float(v.reshape(-1)[ix].astype(np.float64).mean()) for n,v in values.items()}
            gpar=means['P300']-means['P1T'];gstu=means['student-P300']-means['student-P1T']
            buckets.append(dict(bucket=index,count=len(ix),key_min=float(key[ix].min()),key_max=float(key[ix].max()),
                means=means,parent_gap=gpar,student_gap=gstu,retention=gstu/gpar if gpar else None,
                low_denominator=abs(gpar)<.005,
                pre_extra=(means['pre-P1T']-means['P1T'])-(means['pre-P300']-means['P300']),
                A_init=means['student-X72']-means['student-P300']))
        out[kind]=buckets
    return dict(tie_rule='stable flattened sequence/target-position order; equal counts differ by at most one',buckets=out)
