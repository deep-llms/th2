"""Cost-head fitting and validation-only policy selection. LLM bodies stay frozen."""
import hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .causal_cost import CostHead, metadata, decisions, cached_policy
from .window_cf import enumerate_costs, PATH
from .evaluate import cluster_ids


@torch.no_grad()
def cache(model, rows, ids, device):
    if model.policy_depth is not None or model.cost_head is not None:
        raise ValueError('Cache only factual frozen model')
    arrays=enumerate_costs(model,rows,ids,device,'hold')
    features=[]
    for i in range(0,len(ids),8):
        model.intervention=None; model.nll(rows.batch(ids[i:i+8],device),PATH)
        features.append(torch.cat(model.features,1).cpu().numpy())
    arrays['features']=np.concatenate(features)
    return arrays


def fit(arrays, model_config, window, stride, seed, out, device='cpu', epochs=30, resume=False, hidden=128, batch=256):
    out=Path(out)
    if out.exists() and not resume:raise ValueError('Fresh head output required')
    if resume and not (out/'fullstate.pt').exists():raise ValueError('Missing resume state')
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed);head=CostHead(model_config['d'],hidden).to(device)
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
    generator=torch.Generator().manual_seed(seed+227)
    x=torch.as_tensor(arrays['features'],device=device).float()
    meta=metadata(model_config['length'],window,stride,device)[None].expand(len(x),-1,-1)
    targets=torch.as_tensor(arrays['costs'],device=device).reshape(len(x),x.shape[1],4)-torch.as_tensor(arrays['nll'],device=device)[:,None,None]
    x=x.reshape(-1,x.shape[-1]);meta=meta.reshape(-1,6);targets=targets.reshape(-1,4)
    locks={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in arrays.items()}
    recipe=dict(config=model_config,window=window,stride=stride,seed=seed,epochs=epochs,hidden=hidden,batch=batch,cache_hashes=locks)
    start=0
    if resume:
        saved=torch.load(out/'fullstate.pt',map_location='cpu',weights_only=False)
        previous=dict(saved['recipe']);previous['epochs']=epochs
        if previous!=recipe:raise ValueError('Head recipe changed')
        head.load_state_dict(saved['head']);optimizer.load_state_dict(saved['optimizer']);generator.set_state(saved['generator']);torch.set_rng_state(saved['rng']);start=saved['epoch']
        if start>=epochs:raise ValueError('No additional epochs')
    head.train();trace=[]
    for epoch in range(start,epochs):
        order=torch.randperm(len(x),generator=generator);total=0.
        for i in range(0,len(order),batch):
            ix=order[i:i+batch].to(device);optimizer.zero_grad(set_to_none=True)
            loss=F.smooth_l1_loss(100*head(x[ix],meta[ix]),100*targets[ix])
            if not torch.isfinite(loss):raise ValueError('Nonfinite head loss')
            loss.backward();optimizer.step();total+=float(loss.detach())*len(ix)
        trace.append(dict(epoch=epoch+1,loss=total/len(order)))
        state=dict(recipe=recipe,epoch=epoch+1,head=head.state_dict(),optimizer=optimizer.state_dict(),generator=generator.get_state(),rng=torch.get_rng_state())
        tmp=out/'fullstate.tmp';torch.save(state,tmp);tmp.replace(out/'fullstate.pt')
    head.eval();torch.save(dict(recipe=recipe,head=head.state_dict(),trace=trace),out/'head.pt')
    return head,trace


@torch.no_grad()
def predict(head, arrays, length, window, stride, device='cpu'):
    features=torch.as_tensor(arrays['features'],device=device).float();m=metadata(length,window,stride,device)
    return torch.cat([head(features[i:i+32],m[None].expand(len(features[i:i+32]),-1,-1)) for i in range(0,len(features),32)]).cpu()


def select(predicted, arrays, length, window, stride):
    layout=decisions(length,window,stride);routes=torch.as_tensor(arrays['routes']).long();costs=torch.as_tensor(arrays['costs']);nll=torch.as_tensor(arrays['nll'])
    candidates=[]
    for depth,threshold in [(None,0.)]+[(d,t) for d in range(4) for t in (0.,.002,.005)]:
        values,action=cached_policy(predicted,routes,costs,nll,depth,threshold,layout)
        candidates.append(dict(depth=depth,threshold=threshold,nll=float(values.double().mean()),edits=int((action>=0).sum())))
    # Stable tie break: no edit, then earlier depth, then lower threshold.
    index=min(range(len(candidates)),key=lambda i:(candidates[i]['nll'],i))
    return dict(selected=candidates[index],candidates=candidates)


def constant(arrays):
    delta=arrays['costs'].reshape(len(arrays['nll']),-1,4)-arrays['nll'][:,None,None]
    return torch.as_tensor(delta.mean(0,dtype=np.float64),dtype=torch.float32)
