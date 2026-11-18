"""Frozen two-exit evaluation, real causal dispatch, on-node counterfactual replay."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import Rows,sha
from .model import Config,Model,ORDERS
from .refinement import Refinement,canonical_state,sparse_block
from .refinement_head import fit_head
from .token_router import precision
from .evaluate import cluster_ids
from .train import write_json


def check_train(run):
    run=Path(run);done=json.loads((run/'complete.json').read_text());recipe=json.loads((run/'recipe.json').read_text())
    c=recipe['config'];cp=json.loads((run/'checkpoint.json').read_text())
    if done['step']!=c['steps'] or cp['step']!=c['steps'] or done['input_tokens']!=c['steps']*c['batch']*c['model']['length']:raise ValueError('Incomplete training')
    if sha(run/'weights.pt')!=done['weights_sha256'] or sha(run/'fullstate.pt')!=done['fullstate_sha256'] or cp['sha256']!=done['fullstate_sha256']:raise ValueError('Checkpoint changed')
    for p,h in recipe['inputs'].items():
        if sha(p)!=h:raise ValueError('Training inputs changed')
    return done,c


def costs(h,w,x,prefix):
    a=h[:,prefix-1:-1].reshape(-1,h.shape[-1]);y=x[:,prefix:].reshape(-1)
    return torch.cat([F.cross_entropy(F.linear(a[i:i+1024],w).float(),y[i:i+1024],reduction='none') for i in range(0,len(y),1024)]).reshape(len(x),-1)


def summarize(c,ch,random):
    def choose(k):return np.take_along_axis(c,k[...,None],-1)[...,0].mean(1,dtype=np.float64)
    return dict(base=c[:,:,0].mean(1,dtype=np.float64),refine=c[:,:,1].mean(1,dtype=np.float64),learned=choose(ch),random=choose(random),oracle=c.min(-1).mean(1,dtype=np.float64),refine_count=ch.sum(1,dtype=np.int64))


@torch.no_grad()
def dispatch(m,x,head):
    h=canonical_state(m.base,x);a=m.base.norm(h);mask=torch.zeros_like(x,dtype=torch.bool)
    mask[:,m.base.c.prefix-1:-1]=head(a[:,m.base.c.prefix-1:-1]).argmax(-1).bool()
    return m.base.norm(sparse_block(m.extra,h,mask)),mask


@torch.no_grad()
def probe(m,x):
    p=m.base.c.prefix;t=x.shape[1];cut=min(p+3,t-1)
    a,b=m.states(x);changed=x.clone();changed[:,cut:]=(changed[:,cut:]+1)%m.base.c.vocab
    aa,bb=m.states(changed);causal=max(float((a[:,:cut]-aa[:,:cut]).abs().max()),float((b[:,:cut]-bb[:,:cut]).abs().max()))
    mask=torch.arange(t,device=x.device)[None,:].expand_as(x)%3==0
    _,z=m.states(x,mask);dense=torch.where(mask[...,None],b,a);sparse=float((z-dense).abs().max())
    single=m.states(x[:1])[1];batch=float((single-b[:1]).abs().max())
    if causal!=0 or sparse>2e-4 or batch>2e-4:raise ValueError('Production causal/sparse/batch gate '+str((causal,sparse,batch)))
    return dict(causal_max=causal,sparse_hidden_max=sparse,batch_max=batch)


@torch.no_grad()
def collect(m,rows,ids,out,device,batch=8,features=False):
    out=Path(out);out.mkdir();p=m.base.c.prefix;n=len(ids);t=m.base.c.length-p
    c=np.lib.format.open_memmap(out/'costs.npy',mode='w+',dtype='f4',shape=(n,t,2))
    feat=np.lib.format.open_memmap(out/'features.npy',mode='w+',dtype='f4',shape=(n,t,m.base.c.d)) if features else None
    for i in range(0,n,batch):
        x=rows.batch(ids[i:i+batch],device);a,b=m.states(x)
        c[i:i+len(x)]=torch.stack([costs(v,m.base.embed.weight,x,p) for v in (a,b)],-1).cpu().numpy()
        if feat is not None:feat[i:i+len(x)]=a[:,p-1:-1].cpu().numpy()
    c.flush()
    if feat is not None:feat.flush()
    return c,feat


@torch.no_grad()
def latency(m,head,x):
    def sync():
        if x.is_cuda:torch.cuda.synchronize()
    def forward(sparse):
        h=dispatch(m,x,head)[0] if sparse else m.states(x)[1]
        return costs(h,m.base.embed.weight,x,m.base.c.prefix)
    times={}
    for sparse in (False,True):
        for _ in range(2):forward(sparse)
        sync();start=time.perf_counter()
        for _ in range(5):forward(sparse)
        sync();times['sparse' if sparse else 'dense']=(time.perf_counter()-start)/5
    return dict(seconds=times,batch=len(x),tokens=x.shape[1],repeats=5,includes_router_and_lm_head=True,claim='full-context teacher-forced latency only')


def evaluate(config,run,out,device='cuda',head_epochs=8):
    precision();torch.set_num_threads(4);run=Path(run);out=Path(out)
    if out.exists():raise ValueError('Fresh eval required')
    done,c=check_train(run)
    for p,h in config['hashes'].items():
        if sha(p)!=h:raise ValueError('Eval inputs changed '+p)
    state=torch.load(run/'weights.pt',map_location='cpu',weights_only=False);base=Model(Config(**state['config']));arm=c['arm'];seed=c['seed']
    m=base if arm=='C6' else Refinement(base,frozen=arm=='F7');m.load_state_dict(state['model']);m.to(device).eval()
    for param in m.parameters():param.requires_grad_(False)
    out.mkdir(parents=True);write_json(out/'config.json',dict(**config,run=str(run),weights_sha256=done['weights_sha256'],head_epochs=head_epochs))
    ids=np.load(config['indices']);rows={d:Rows(s,c['model']['length']) for d,s in config['stems'].items()};start=time.monotonic();proof={};valid_rate=None
    if arm!='C6':
        proof=probe(m,rows['fit'].batch(ids['valid'][:2],device))
        fit,ff=collect(m,rows['fit'],ids['fit'],out/'fit',device,features=True);val,vf=collect(m,rows['fit'],ids['valid'],out/'valid',device,features=True)
        head=fit_head(ff,fit,vf,val,seed+982,out/'head',device,epochs=head_epochs)
        with torch.no_grad():
            vc=np.concatenate([head(torch.as_tensor(np.array(vf[i:i+8]),device=device)).argmax(-1).cpu().numpy() for i in range(0,len(vf),8)])
        valid_rate=float(vc.mean());np.save(out/'valid/choices.npy',vc)
        del ff,vf,fit,val
    for domain in ('dclm','fwe'):
        selected=ids[domain];dest=out/domain
        if arm=='C6':
            dest.mkdir();values=[]
            with torch.no_grad():
                for i in range(0,len(selected),8):values.append(m.nll(rows[domain].batch(selected[i:i+8],device),ORDERS[0]).double().cpu().numpy())
            summary=dict(dense=np.concatenate(values));np.save(dest/'costs.npy',summary['dense'])
        else:
            costs_array,_=collect(m,rows[domain],selected,dest,device);ch=[];actual=[];maxerr=0.;kv_tokens=0
            with torch.no_grad():
                for i in range(0,len(selected),8):
                    x=rows[domain].batch(selected[i:i+8],device);h,mask=dispatch(m,x,head);choices=mask[:,m.base.c.prefix-1:-1].long().cpu().numpy();ch.append(choices)
                    v=costs(h,m.base.embed.weight,x,m.base.c.prefix).cpu().numpy();actual.append(v.mean(1,dtype=np.float64))
                    expected=np.take_along_axis(costs_array[i:i+len(x)],choices[...,None],-1)[...,0];maxerr=max(maxerr,float(np.max(np.abs(v-expected))));kv_tokens+=int(mask.any(1).sum())*x.shape[1]
            if maxerr>2e-4:raise ValueError('Actual dispatch token NLL mismatch '+str(maxerr))
            ch=np.concatenate(ch);random=(np.random.default_rng(889+seed).random(ch.shape)<valid_rate).astype(np.int64)
            np.save(dest/'choices.npy',ch);np.save(dest/'random.npy',random);np.save(dest/'actual.npy',np.concatenate(actual))
            summary=summarize(costs_array,ch,random);summary['actual_sparse']=np.concatenate(actual)
            proof[domain]=dict(sparse_token_max_error=maxerr,base_block_tokens=6*len(selected)*c['model']['length'],extra_query_tokens=int(ch.sum()),extra_kv_tokens=kv_tokens,router_tokens=int(ch.size),latency=latency(m,head,rows[domain].batch(selected[:8],device)))
        np.savez(dest/'summary.npz',ids=selected,clusters=cluster_ids(rows[domain],selected),**summary)
    check_train(run)
    for p,h in config['hashes'].items():
        if sha(p)!=h:raise ValueError('Input mutation')
    write_json(out/'complete.json',dict(status='complete',arm=arm,seed=seed,seconds=time.monotonic()-start,validation_refine_rate=valid_rate,probes=proof,files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--run',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');a=p.parse_args();evaluate(json.loads(Path(a.config).read_text()),a.run,a.out,a.device)
