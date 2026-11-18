"""Frozen-backbone, full-information routing; test is accessed after head selection."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from .model import Config,Model,Router,ORDERS,router_loss
from .data import Rows,sha
from .train import amp,write_json

@torch.no_grad()
def collect(model,rows,ids,device,batch=16,all_orders=True):
    costs=[];features=[];model.eval()
    for i in range(0,len(ids),batch):
        x=rows.batch(ids[i:i+batch],device)
        with amp(device):
            features.append(model.features(x).cpu().numpy())
            costs.append(torch.stack([model.nll(x,p) for p in (ORDERS if all_orders else ORDERS[:1])],-1).cpu().numpy())
    costs,features=np.concatenate(costs),np.concatenate(features)
    if not np.isfinite(costs).all() or not np.isfinite(features).all():raise ValueError('Nonfinite evaluation')
    return costs,features

def fit_router(fit_features,fit_cost,dev_features,dev_cost,seed,device,epochs=30):
    torch.manual_seed(seed);router=Router(fit_features.shape[1]).to(device)
    opt=torch.optim.AdamW(router.parameters(),lr=.001,weight_decay=.01)
    x=torch.as_tensor(fit_features,device=device);y=torch.as_tensor(fit_cost,device=device)
    xd=torch.as_tensor(dev_features,device=device);yd=torch.as_tensor(dev_cost,device=device)
    gen=torch.Generator().manual_seed(seed+2701);best=float('inf');best_state=None;trace=[]
    for epoch in range(epochs):
        router.train();order=torch.randperm(len(x),generator=gen)
        for ix in order.split(256):
            ix=ix.to(device);opt.zero_grad(set_to_none=True)
            loss=router_loss(router(x[ix]),y[ix]);loss.backward();opt.step()
        router.eval()
        with torch.no_grad():
            chosen=router(xd).argmax(-1);value=float(yd[torch.arange(len(yd),device=device),chosen].mean())
        trace.append(dict(epoch=epoch+1,dev_selected_nll=value))
        if value<best:best=value;best_state=copy.deepcopy(router.state_dict());best_epoch=epoch+1
    router.load_state_dict(best_state);return router,dict(epoch=best_epoch,dev_nll=best,trace=trace)

def summaries(costs,choices,constant):
    return dict(canonical=costs[:,0],uniform=costs.mean(1),constant=costs[:,constant],learned=costs[np.arange(len(costs)),choices],oracle=costs.min(1))

def cluster_ids(rows,ids):
    # Each original packed tuning/final row belongs to its source-document partition.
    starts=np.asarray(rows.starts);totals=starts.sum(1,dtype=np.int64)
    return np.cumsum(totals)[ids]-totals[ids]+starts[ids,0]

def paired_ci(delta,clusters,reps=2000):
    _,inv=np.unique(clusters,return_inverse=True)
    sums=np.bincount(inv,weights=delta);counts=np.bincount(inv)
    rng=np.random.default_rng(901);samples=[]
    for _ in range(reps):
        ix=rng.integers(len(sums),size=len(sums));samples.append(sums[ix].sum()/counts[ix].sum())
    return dict(mean=float(delta.mean()),lo=float(np.quantile(samples,.025)),hi=float(np.quantile(samples,.975)),documents=len(sums),sequences=len(delta))

def evaluate(config_path,out,device='cuda'):
    cfg=json.loads(Path(config_path).read_text());out=Path(out)
    if out.exists():raise ValueError('Fresh evaluation output required')
    out.mkdir(parents=True);torch.set_num_threads(4);start=time.monotonic()
    run=Path(cfg['run']);done=json.loads((run/'complete.json').read_text())
    if sha(run/'weights.pt')!=done['weights_sha256']:raise ValueError('Weights hash mismatch')
    state=torch.load(run/'weights.pt',map_location='cpu',weights_only=False)
    model=Model(Config(**state['config'])).to(device);model.load_state_dict(state['model']);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    # Validate all declared inputs, including held-out data, without reading their targets into fitting.
    for name,want in cfg['input_hashes'].items():
        if sha(name)!=want:raise ValueError('Evaluation input hash mismatch: '+name)
    indices=np.load(cfg['indices'],allow_pickle=False)
    all_orders=done['arm']!='D';selection=None;router=None;constant=0
    if all_orders:
        fit,fx=collect(model,Rows(cfg['fit'],model.c.length),indices['fit'],device)
        dev,dx=collect(model,Rows(cfg['dev'],model.c.length),indices['dev'],device)
        constant=int(fit.mean(0).argmin())
        router,selection=fit_router(fx,fit,dx,dev,done['seed'],device)
        torch.save(router.state_dict(),out/'router.pt')
        write_json(out/'selection.json',dict(constant=constant,selection=selection))
        np.savez_compressed(out/'fit_dev_costs.npz',fit=fit,dev=dev,fit_ids=indices['fit'],dev_ids=indices['dev'])
    # This is the first test-label evaluation; choices above are now immutable.
    testrows=Rows(cfg['test'],model.c.length);ids=indices['test'];costs,features=collect(model,testrows,ids,device,all_orders=all_orders)
    if router is not None:
        with torch.no_grad():choices=router(torch.as_tensor(features,device=device)).argmax(-1).cpu().numpy()
        metrics=summaries(costs,choices,constant)
    else:choices=np.zeros(len(ids),dtype=np.int64);metrics=dict(canonical=costs[:,0])
    clusters=cluster_ids(testrows,ids)
    np.savez_compressed(out/'test_costs.npz',costs=costs,choices=choices,ids=ids,clusters=clusters,**metrics)
    report=dict(arm=done['arm'],seed=done['seed'],weights_sha256=done['weights_sha256'],constant=constant,selection=selection,
                metrics={k:float(v.mean()) for k,v in metrics.items()},sequences=len(ids),scored_tokens=len(ids)*(model.c.length-model.c.prefix),
                seconds=time.monotonic()-start,flags=['conditional_suffix_LM','two_seeds','unequal_training_block_compute','weights_node_only'])
    if router is not None:
        report['learned_minus_constant']=paired_ci(metrics['learned']-metrics['constant'],clusters)
        report['learned_minus_uniform']=paired_ci(metrics['learned']-metrics['uniform'],clusters)
    write_json(out/'report.json',report)
    write_json(out/'complete.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()},weights_sha256=done['weights_sha256']))
    print(json.dumps(report),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');a=p.parse_args();evaluate(a.config,a.out,a.device)
if __name__=='__main__':main()
