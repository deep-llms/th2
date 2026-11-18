"""Frozen warm/local-order continuation scorer; no target may enter the router."""
import argparse,collections,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .model import Config,Model,Router,ORDERS
from .local_policy import LocalRouter,LOCAL_ORDERS
from .train import amp,write_json
from .data import sha
from eval.a2_request_bridge import digest


def prepare_rows(requests,length=512):
    longest={}
    for r in requests:
        ctx=tuple(r['context'] or [0]);target=r['continuation']
        if not target or len(target)>length:raise ValueError('Invalid continuation length')
        longest[ctx]=max(longest.get(ctx,0),len(target))
    rows=[]
    for i,r in enumerate(requests):
        ctx=tuple(r['context'] or [0]);target=list(r['continuation'])
        kept=list(ctx[-(length+1-longest[ctx]):]);tokens=(kept+target)[:-1]
        rows.append(dict(index=i,context=kept,tokens=tokens,target=target,truncated=len(kept)<len(ctx)))
    return rows


@torch.no_grad()
def select_routes(model,router,contexts,device,batch_size=32):
    if router is None:return {c:0 for c in contexts}
    # Group actual prefix lengths: no EOS padding used to obtain router features.
    groups=collections.defaultdict(list)
    for c in contexts:groups[min(model.c.prefix,len(c))].append(c)
    result={}
    for n,group in groups.items():
        for start in range(0,len(group),batch_size):
            part=group[start:start+batch_size];x=torch.tensor([c[:n] for c in part],device=device)
            with amp(device):choices=router(model.features(x)).argmax(-1).tolist()
            result.update(zip(part,choices))
    return result


@torch.no_grad()
def score(model,router,requests,orders,device='cpu',batch_size=16):
    if batch_size<1:raise ValueError('Invalid batch size')
    model.eval()
    if router is not None:router.eval()
    rows=prepare_rows(requests,model.c.length);contexts=sorted({tuple(r['context']) for r in rows})
    routes=select_routes(model,router,contexts,device,batch_size)
    groups=collections.defaultdict(list)
    for row in rows:groups[routes[tuple(row['context'])]].append(row)
    scores=[None]*len(rows);choices=[None]*len(rows)
    for route,group in groups.items():
        group.sort(key=lambda r:-len(r['tokens']))
        for start in range(0,len(group),batch_size):
            part=group[start:start+batch_size];ids=torch.zeros((len(part),len(part[0]['tokens'])),device=device,dtype=torch.long)
            for i,r in enumerate(part):ids[i,:len(r['tokens'])]=torch.tensor(r['tokens'],device=device)
            with amp(device):h=model.hidden(ids,orders[route])
            for i,r in enumerate(part):
                a=len(r['context'])-1;b=a+len(r['target'])
                with amp(device):logits=F.linear(h[i,a:b],model.embed.weight)
                logits=logits.float();target=torch.tensor(r['target'],device=device)
                value=logits.log_softmax(-1).gather(1,target[:,None]).sum()
                if not torch.isfinite(value):raise ValueError('Nonfinite score')
                scores[r['index']]=[float(value),bool(torch.equal(logits.argmax(-1),target))];choices[r['index']]=route
    return scores,dict(choices=choices,context_routes={str(k):sum(v==k for v in routes.values()) for k in range(len(orders))},unique_contexts=len(contexts),truncated_requests=sum(r['truncated'] for r in rows),short_prefix_requests=sum(len(r['context'])<model.c.prefix for r in rows),common_context_truncation=True)


def load(entry,device):
    for p,w in entry['hashes'].items():
        if sha(p)!=w:raise ValueError('Checkpoint/hash changed '+p)
    state=torch.load(entry['weights'],map_location='cpu',weights_only=False)
    model=Model(Config(**state['config'])).to(device);model.load_state_dict(state['model'])
    router=None;orders=LOCAL_ORDERS if entry['study']=='local' else ORDERS
    if entry.get('router'):
        router=Router(model.c.d).to(device);router.load_state_dict(torch.load(entry['router'],map_location='cpu',weights_only=False))
    elif entry['arm'] in ('CF','PG'):
        router=LocalRouter(model.c.d).to(device);router.load_state_dict(state['router'])
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model,router,orders


def evaluate(plan_path,name,out,device='cuda'):
    plan=json.loads(Path(plan_path).read_text());entry=next(x for x in plan['models'] if x['name']==name)
    if sha(plan['requests'])!=plan['requests_sha256']:raise ValueError('Requests changed')
    prepared=json.loads(Path(plan['requests']).read_text());requests=prepared['requests'];out=Path(out)
    if out.exists():raise ValueError('Fresh output required')
    model,router,orders=load(entry,device);torch.set_num_threads(4);start=time.monotonic()
    scores,details=score(model,router,requests,orders,device,plan.get('batch_size',16));out.mkdir(parents=True)
    write_json(out/'scores.json',dict(scores=scores,likelihood_requests=len(scores),precision='fp32_parameters_bf16_cuda_autocast',max_length=512,prepared_sha256=digest(prepared),request_sha256=prepared['request_sha256']))
    write_json(out/'details.json',dict(**details,seconds=time.monotonic()-start,name=name,checkpoint_hashes=entry['hashes']))
    write_json(out/'complete.json',dict(name=name,files={p.name:sha(p) for p in out.iterdir()},requests_sha256=plan['requests_sha256']))


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--name',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');a=p.parse_args();evaluate(a.plan,a.name,a.out,a.device)
if __name__=='__main__':main()
