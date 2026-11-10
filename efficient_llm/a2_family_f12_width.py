"""Native SwiGLU neuron selection; attention and every decoder layer stay intact."""
import numpy as np
import torch
from torch import nn
from efficient_llm.a2.model import autocast,assert_native

def calibration_indices(total,count=1024,seed=2031):
    if total<count:raise ValueError('Insufficient heal sequences')
    return np.random.default_rng(seed).choice(total,count,replace=False).tolist()

def select_neurons(means,keep):
    a=np.asarray(means,dtype=np.float64)
    if a.ndim!=2 or not np.isfinite(a).all() or not 0<keep<=a.shape[1]:raise ValueError('Invalid neuron importance')
    # Rank by magnitude, break exact ties by original index; serialize original
    # order so retaining every neuron is bitwise identical to the parent.
    return [sorted(np.lexsort((np.arange(len(row)),-row))[:keep].tolist()) for row in a]

@torch.no_grad()
def slice_width(model,masks):
    layers=model.model.layers
    if len(masks)!=len(layers) or len({len(x) for x in masks})!=1:raise ValueError('Invalid per-layer masks')
    for layer,keep in zip(layers,masks,strict=True):
        mlp=layer.mlp;old=mlp.gate_proj.out_features
        if sorted(set(keep))!=keep or not keep or keep[-1]>=old or keep[0]<0:raise ValueError('Invalid neuron indices')
        ix=torch.tensor(keep,device=mlp.gate_proj.weight.device)
        for key in ('gate_proj','up_proj','down_proj'):
            src=getattr(mlp,key);dim=1 if key=='down_proj' else 0
            shape=(len(keep),src.out_features) if dim==1 else (src.in_features,len(keep))
            dst=nn.Linear(*shape,bias=src.bias is not None,device=src.weight.device,dtype=src.weight.dtype)
            dst.weight.copy_(src.weight.index_select(dim,ix))
            if src.bias is not None:dst.bias.copy_(src.bias if dim==1 else src.bias[ix])
            setattr(mlp,key,dst)
        mlp.intermediate_size=len(keep)
    model.config.intermediate_size=len(masks[0])
    model.config.a2_width_masks=masks
    model.config.a2_kept_layers=list(range(len(layers)))
    assert_native(model)
    return sum(p.numel() for p in model.parameters())-model.get_input_embeddings().weight.numel()

@torch.no_grad()
def importance(model,rows,indices,rank=0,world=1):
    device=next(model.parameters()).device;width=model.config.intermediate_size
    sums=torch.zeros((len(model.model.layers),width),device=device,dtype=torch.float64)
    counts=torch.zeros(len(model.model.layers),device=device,dtype=torch.int64);hooks=[]
    def hook(i):
        def capture(module,args):
            x=args[0];sums[i].add_(x.float().abs().sum(tuple(range(x.ndim-1)),dtype=torch.float64));counts[i]+=x.numel()//width
        return capture
    for i,layer in enumerate(model.model.layers):hooks.append(layer.mlp.down_proj.register_forward_pre_hook(hook(i)))
    try:
        model.eval()
        for j in indices[rank::world]:
            ids=rows[j]['input_ids'].unsqueeze(0).to(device)
            with autocast(device):model.model(ids,use_cache=False,return_dict=True)
    finally:
        for handle in hooks:handle.remove()
    if world>1:
        torch.distributed.all_reduce(sums);torch.distributed.all_reduce(counts)
    expected=len(indices)*len(rows[indices[0]]['input_ids'])
    if not bool(torch.all(counts==expected)):raise ValueError('Importance token coverage mismatch')
    return (sums/counts[:,None]).cpu().numpy(),counts.cpu().tolist()
