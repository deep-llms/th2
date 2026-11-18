"""Final checkpoint evaluation; no parameter or checkpoint selection on test."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .data import Rows,sha
from .model import Config,Model
from .local_policy import LocalRouter,LOCAL_ORDERS
from .train import amp,write_json
from .evaluate import cluster_ids,paired_ci


@torch.no_grad()
def collect(model, router, rows, ids, device, batch=16, all_orders=True):
    model.eval()
    if router is not None:router.eval()
    costs=[];choices=[]
    for start in range(0,len(ids),batch):
        x=rows.batch(ids[start:start+batch],device)
        with amp(device):
            costs.append(torch.stack([model.nll(x,p) for p in (LOCAL_ORDERS if all_orders else LOCAL_ORDERS[:1])],-1).cpu().numpy())
            if router is not None:choices.append(router(model.features(x)).argmax(-1).cpu().numpy())
            else:choices.append(np.zeros(len(x),dtype=np.int64))
    costs=np.concatenate(costs)
    if not np.isfinite(costs).all():raise ValueError('Nonfinite eval')
    return costs,np.concatenate(choices)


def evaluate(config,out,device='cuda'):
    out=Path(out)
    if out.exists():raise ValueError('Fresh evaluation output required')
    c=dict(config);run=Path(c['run']);done=json.loads((run/'complete.json').read_text())
    if sha(run/'weights.pt')!=done['weights_sha256']:raise ValueError('Checkpoint hash mismatch')
    for p,want in c['input_hashes'].items():
        if sha(p)!=want:raise ValueError('Input changed '+p)
    torch.set_num_threads(4);start=time.monotonic()
    state=torch.load(run/'weights.pt',map_location='cpu',weights_only=False)
    model=Model(Config(**state['config'])).to(device);model.load_state_dict(state['model'])
    router=LocalRouter(model.c.d).to(device) if state['router'] is not None else None
    if router is not None:router.load_state_dict(state['router'])
    for p in model.parameters():p.requires_grad_(False)
    indices=np.load(c['indices'],allow_pickle=False);all_orders=done['arm']!='C';constant=0
    out.mkdir(parents=True)
    if all_orders:
        fit,_=collect(model,router,Rows(c['fit'],model.c.length),indices['fit'],device,all_orders=True)
        constant=int(fit.mean(0).argmin())
        np.savez_compressed(out/'fit_costs.npz',costs=fit,ids=indices['fit'])
    reports={}
    for domain in ('dclm','fwe'):
        rows=Rows(c[domain],model.c.length);ids=indices[domain]
        costs,choices=collect(model,router,rows,ids,device,all_orders=all_orders)
        metrics=dict(canonical=costs[:,0])
        if all_orders:
            metrics.update(constant=costs[:,constant],uniform=costs.mean(1),oracle=costs.min(1))
            # U's deployment comparator is query-independent, with the actual
            # fixed training mixture; no noisy single route sampling on test.
            metrics['mixture']=costs@np.array([.79,.07,.07,.07],dtype=np.float32)
        if router is not None:metrics['learned']=costs[np.arange(len(costs)),choices]
        clusters=cluster_ids(rows,ids)
        np.savez_compressed(out/f'{domain}_costs.npz',costs=costs,choices=choices,ids=ids,clusters=clusters,**metrics)
        reports[domain]=dict(metrics={k:float(v.mean()) for k,v in metrics.items()},
                             sequences=len(ids),scored_tokens=len(ids)*(model.c.length-model.c.prefix))
        if router is not None:
            reports[domain]['choices']=np.bincount(choices,minlength=4).tolist()
            reports[domain]['learned_minus_constant']=paired_ci(metrics['learned']-metrics['constant'],clusters)
    report=dict(arm=done['arm'],seed=done['seed'],weights_sha256=done['weights_sha256'],
                constant=constant,domains=reports,seconds=time.monotonic()-start)
    write_json(out/'report.json',report)
    write_json(out/'complete.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()},weights_sha256=done['weights_sha256']))
    print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();evaluate(json.loads(Path(a.config).read_text()),a.out,a.device)
if __name__=='__main__':main()
