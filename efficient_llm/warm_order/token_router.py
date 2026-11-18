"""Frozen causal order experts, token-wise full-information routing. No cheap KV claim."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .model import Config, Model, Router, ORDERS
from .data import Rows, sha
from .train import write_json
from .evaluate import cluster_ids


def precision():
    torch.set_float32_matmul_precision('highest')
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


@torch.no_grad()
def stem_features(model, x):
    pos = torch.arange(x.shape[1], device=x.device).expand_as(x)
    h = model.stem(model.embed(x), pos, torch.zeros_like(x))
    return h[:, model.c.prefix-1:-1].float().detach()


@torch.no_grad()
def token_cost(model, x, path):
    h = model.hidden(x, path)[:, model.c.prefix-1:-1].reshape(-1, model.c.d)
    y = x[:, model.c.prefix:].reshape(-1)
    loss = [F.cross_entropy(F.linear(h[i:i+1024], model.embed.weight).float(),
                           y[i:i+1024], reduction='none') for i in range(0,len(y),1024)]
    return torch.cat(loss).reshape(len(x), -1)


def expected_loss(logits, costs):
    c = costs.detach(); c = c-c.mean(-1, keepdim=True)
    return (logits.softmax(-1)*c).sum(-1).mean()


def fit_head(features, costs, valid_features, valid_costs, seed, out, device,
             epochs=8, batch=4096, stop_after=None, resume=False):
    """Fullstate epoch-boundary resume; private permutation generator and fixed schedule."""
    out=Path(out)
    def digest(array):
        raw=memoryview(np.ascontiguousarray(array)).cast('B');h=hashlib.sha256()
        for i in range(0,len(raw),8*1024*1024):h.update(raw[i:i+8*1024*1024])
        return h.hexdigest()
    recipe=dict(input_sha256=[digest(a) for a in (features,costs,valid_features,valid_costs)],seed=seed,epochs=epochs,batch=batch,lr=.001,weight_decay=.01,
        features=list(features.shape),costs=list(costs.shape),valid=list(valid_features.shape))
    if resume:
        state=torch.load(out/'fullstate.pt',map_location='cpu',weights_only=False)
        if state['recipe']!=recipe:raise ValueError('Head recipe changed')
    elif out.exists():raise ValueError('Fresh head output required')
    else:out.mkdir(parents=True)
    torch.manual_seed(seed);head=Router(features.shape[-1]).to(device)
    opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
    gen=torch.Generator().manual_seed(seed+1);epoch0=0;trace=[];best=float('inf');best_state=None
    x=torch.as_tensor(np.asarray(features).reshape(-1,features.shape[-1]).copy(),device=device)
    c=torch.as_tensor(np.asarray(costs).reshape(-1,8).copy(),device=device)
    vx=torch.as_tensor(np.asarray(valid_features).reshape(-1,features.shape[-1]).copy(),device=device)
    vc=torch.as_tensor(np.asarray(valid_costs).reshape(-1,8).copy(),device=device)
    if resume:
        head.load_state_dict(state['head']);opt.load_state_dict(state['optimizer']);gen.set_state(state['generator'])
        epoch0=state['epoch'];trace=state['trace'];best=state['best'];best_state=state['best_head']
    for epoch in range(epoch0,epochs):
        head.train();order=torch.randperm(len(x),generator=gen)
        for ix in order.split(batch):
            ix=ix.to(device);opt.zero_grad(set_to_none=True)
            loss=expected_loss(head(x[ix]),c[ix]);loss.backward();opt.step()
        head.eval()
        with torch.no_grad():
            total=0.
            for i in range(0,len(vx),batch):
                ch=head(vx[i:i+batch]).argmax(-1)
                total+=float(vc[i:i+batch].gather(1,ch[:,None]).double().sum())
        value=total/len(vx);trace.append(dict(epoch=epoch+1,validation_hard_nll=value))
        if value<best:best=value;best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        full=dict(recipe=recipe,epoch=epoch+1,head=head.state_dict(),optimizer=opt.state_dict(),
                  generator=gen.get_state(),trace=trace,best=best,best_head=best_state)
        torch.save(full,out/'fullstate.tmp');(out/'fullstate.tmp').replace(out/'fullstate.pt')
        write_json(out/'progress.json',dict(epoch=epoch+1,best=best,trace=trace))
        if stop_after is not None and epoch+1>=stop_after:return None
    head.load_state_dict(best_state);head.eval();torch.save(best_state,out/'router.pt')
    write_json(out/'complete.json',dict(recipe=recipe,trace=trace,best=best,
        best_epoch=min(trace,key=lambda r:r['validation_hard_nll'])['epoch'],
        router_sha256=sha(out/'router.pt'),fullstate_sha256=sha(out/'fullstate.pt')))
    return head


def id_selector(tokens,costs,vocab,smoothing=32.):
    t=np.asarray(tokens).reshape(-1);c=np.asarray(costs).reshape(-1,8)
    sums=np.zeros((vocab,8),dtype=np.float64);np.add.at(sums,t,c)
    counts=np.bincount(t,minlength=vocab)
    means=(sums+smoothing*c.mean(0,dtype=np.float64))/(counts[:,None]+smoothing)
    return means.argmin(-1).astype(np.uint8)


def summarize(costs, choices, constant):
    c=np.asarray(costs); n,t,k=c.shape
    def pick(ch):return np.take_along_axis(c,np.asarray(ch)[...,None],axis=2)[...,0].mean(1,dtype=np.float64)
    result=dict(canonical=c[:,:,0].mean(1,dtype=np.float64))
    if k==8:
        low=c.min(-1).astype(np.float64)
        ensemble=low-np.log(np.exp(-(c.astype(np.float64)-low[:,:,None])).mean(-1))
        result.update(constant=c[:,:,constant].mean(1,dtype=np.float64),
                      uniform=c.mean((1,2),dtype=np.float64),oracle=low.mean(1),ensemble=ensemble.mean(1))
        for key,ch in choices.items():result[key]=pick(ch)
    return result


@torch.no_grad()
def probe(model,x):
    """Production exact-precision target-shift, causal state and batching gates."""
    p=model.c.prefix;h=stem_features(model,x);changed=x.clone();cut=p+2
    changed[:,cut:]=(changed[:,cut:]+1)%model.c.vocab
    altered=stem_features(model,changed)
    causal=float((h[:,:cut-(p-1)]-altered[:,:cut-(p-1)]).abs().max())
    if causal>2e-5:raise ValueError('Future information leaked into token features')
    errors=[];batch_errors=[]
    for path in ORDERS:
        c=token_cost(model,x,path)
        errors.append(float((c.mean(1)-model.nll(x,path)).abs().max()))
        batch_errors.append(float((c[:1]-token_cost(model,x[:1],path)).abs().max()))
    if max(errors)>2e-5 or max(batch_errors)>2e-4:raise ValueError('FP32 token/sequence/batch parity failed')
    r=Router(model.c.d).to(x.device).eval()
    route_error=float((r(h[:,:cut-(p-1)])-r(altered[:,:cut-(p-1)])).abs().max())
    if route_error>2e-5:raise ValueError('Future route leakage')
    return dict(causal_feature_error=causal,causal_router_error=route_error,
                sequence_errors=errors,batch_errors=batch_errors,precision='fp32_no_autocast_no_tf32')


@torch.no_grad()
def cache(model,rows,ids,out,device,batch=8):
    out=Path(out);out.mkdir()
    n=len(ids);t=model.c.length-model.c.prefix
    f=np.lib.format.open_memmap(out/'features.npy',mode='w+',dtype=np.float32,shape=(n,t,model.c.d))
    c=np.lib.format.open_memmap(out/'costs.npy',mode='w+',dtype=np.float32,shape=(n,t,8))
    tokens=np.lib.format.open_memmap(out/'tokens.npy',mode='w+',dtype=np.int64,shape=(n,t))
    for i in range(0,n,batch):
        x=rows.batch(ids[i:i+batch],device);j=i+len(x)
        f[i:j]=stem_features(model,x).cpu().numpy()
        c[i:j]=torch.stack([token_cost(model,x,p) for p in ORDERS],-1).cpu().numpy()
        tokens[i:j]=x[:,model.c.prefix-1:-1].cpu().numpy()
        write_json(out/'progress.json',dict(rows=j,total=n,utc=time.time()))
    for a in (f,c,tokens):a.flush()
    if not np.isfinite(f).all() or not np.isfinite(c).all():raise ValueError('Nonfinite cache')
    write_json(out/'complete.json',dict(files={p.name:sha(p) for p in out.glob('*.npy')}))
    return f,c,tokens


@torch.no_grad()
def evaluate_domain(model,head,oldhead,lookup,constant,rows,ids,out,device,batch=8):
    out=Path(out);out.mkdir();n=len(ids);t=model.c.length-model.c.prefix;k=8 if head is not None else 1
    c=np.lib.format.open_memmap(out/'costs.npy',mode='w+',dtype=np.float32,shape=(n,t,k))
    choices={key:np.lib.format.open_memmap(out/(key+'.npy'),mode='w+',dtype=np.uint8,shape=(n,t))
             for key in ('learned','prefix','token_id')} if head is not None else {}
    for i in range(0,n,batch):
        x=rows.batch(ids[i:i+batch],device);j=i+len(x)
        c[i:j]=torch.stack([token_cost(model,x,p) for p in ORDERS[:k]],-1).cpu().numpy()
        if head is not None:
            choices['learned'][i:j]=head(stem_features(model,x)).argmax(-1).cpu().numpy()
            prefix=oldhead(model.features(x)).argmax(-1).cpu().numpy()
            choices['prefix'][i:j]=prefix[:,None]
            choices['token_id'][i:j]=lookup[x[:,model.c.prefix-1:-1].cpu().numpy()]
        write_json(out/'progress.json',dict(rows=j,total=n,utc=time.time()))
    c.flush()
    for v in choices.values():v.flush()
    if not np.isfinite(c).all():raise ValueError('Nonfinite final costs')
    arrays=summarize(c,choices,constant);arrays.update(ids=ids,clusters=cluster_ids(rows,ids))
    counts={key:np.stack([(v==a).sum(1) for a in range(8)],1) for key,v in choices.items()}
    arrays.update({key+'_counts':value for key,value in counts.items()})
    arrays.update({key+'_switches':(v[:,1:]!=v[:,:-1]).sum(1) for key,v in choices.items()})
    np.savez_compressed(out/'summary.npz',**arrays)
    metrics={key:float(value.mean()) for key,value in arrays.items() if value.ndim==1 and key not in ('ids','clusters') and not key.endswith('switches')}
    write_json(out/'complete.json',dict(metrics=metrics,sequences=n,targets=n*t,constant=constant,
        choice_counts={key:value.sum(0).tolist() for key,value in counts.items()},
        files={p.name:sha(p) for p in out.iterdir() if p.suffix in ('.npy','.npz')}))


def run(config,out,device='cuda'):
    out=Path(out)
    if out.exists():raise ValueError('Fresh experiment output required; do not blindly retry')
    precision();torch.set_num_threads(4);started=time.monotonic()
    for p,want in config['hashes'].items():
        if sha(p)!=want:raise ValueError('Changed input '+p)
    out.mkdir(parents=True);write_json(out/'config.json',config)
    state=torch.load(config['weights'],map_location='cpu',weights_only=False)
    model=Model(Config(**state['config'])).to(device);model.load_state_dict(state['model']);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    ids=np.load(config['indices'],allow_pickle=False);rows=Rows(config['stems']['fit'],model.c.length)
    write_json(out/'probe.json',probe(model,rows.batch(ids['fit'][:2],device)))
    head=None;oldhead=None;lookup=None;constant=0;cache_seconds=0.;fit_seconds=0.
    if config['arm']!='D':
        oldhead=Router(model.c.d).to(device);oldhead.load_state_dict(torch.load(config['router'],map_location=device,weights_only=False));oldhead.eval()
        t0=time.monotonic();f,c,tokens=cache(model,rows,ids['fit'],out/'fit',device,config.get('batch',8))
        vf,vc,_=cache(model,rows,ids['valid'],out/'valid',device,config.get('batch',8));cache_seconds=time.monotonic()-t0
        constant=int(c.mean((0,1),dtype=np.float64).argmin());lookup=id_selector(tokens,c,model.c.vocab)
        np.save(out/'token_id_lookup.npy',lookup);t0=time.monotonic()
        head=fit_head(f,c,vf,vc,55200+config['seed'],out/'head',device,epochs=config.get('epochs',8),batch=config.get('head_batch',4096));fit_seconds=time.monotonic()-t0
        del f,c,tokens,vf,vc
    write_json(out/'selection.json',dict(constant=constant,test_unseen=True,cache_seconds=cache_seconds,head_seconds=fit_seconds))
    for domain in ('dclm','fwe'):
        evaluate_domain(model,head,oldhead,lookup,constant,Rows(config['stems'][domain],model.c.length),ids[domain],out/domain,device,config.get('batch',8))
    for p,want in config['hashes'].items():
        if sha(p)!=want:raise ValueError('Input mutated '+p)
    forwards=sum(len(ids[d]) for d in ('dclm','fwe'))*(8 if head is not None else 1)
    if head is not None:forwards+=8*(len(ids['fit'])+len(ids['valid']))
    proof=dict(name=config['name'],arm=config['arm'],seed=config['seed'],seconds=time.monotonic()-started,
        full_path_sequence_forwards=forwards+40,extra_stem_sequence_forwards=(len(ids['fit'])+len(ids['valid'])+2*sum(len(ids[d]) for d in ('dclm','fwe')) if head is not None else 0)+4,
        compute_note='Counts include production parity probe40 full paths and4 extra stems; head updates reported separately.',
        frozen_weights_sha256=config['hashes'][config['weights']],status='complete',
        files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='progress.json'})
    write_json(out/'complete.json',proof);print(json.dumps(proof),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.out,a.device)
if __name__=='__main__':main()
