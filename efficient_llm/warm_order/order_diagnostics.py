"""Frozen component swaps and canonical deep causal features; prior artifacts immutable."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import Rows,sha
from .model import Config,Model,ORDERS
from .train import write_json
from .evaluate import cluster_ids
from .token_router import precision,fit_head,token_cost

PATHS=[(0,1,2,3),(1,0,2,3),(0,2,1,3),(0,1,3,2)]

@torch.no_grad()
def hidden(model,x,kind,path):
    if kind not in ('attention','ffn','block'):raise ValueError('Unknown component')
    pos=torch.arange(x.shape[1],device=x.device).expand_as(x);seg=torch.zeros_like(x)
    h=model.stem(model.embed(x),pos,seg)
    for j in range(4):
        a=model.middle[path[j] if kind in ('attention','block') else j]
        f=model.middle[path[j] if kind in ('ffn','block') else j]
        h=a.attention(h,pos,seg);h=h+f.mlp(f.post_attention_layernorm(h))
    return model.norm(model.coda(h,pos,seg))

@torch.no_grad()
def deep_features(model,x):
    return model.hidden(x,ORDERS[0])[:,model.c.prefix-1:-1].float()

@torch.no_grad()
def costs(model,x,kind,path):
    h=hidden(model,x,kind,path)[:,model.c.prefix-1:-1].reshape(-1,model.c.d)
    y=x[:,model.c.prefix:].reshape(-1)
    return torch.cat([F.cross_entropy(F.linear(h[i:i+1024],model.embed.weight).float(),y[i:i+1024],reduction='none') for i in range(0,len(y),1024)]).reshape(len(x),-1)

@torch.no_grad()
def probe(model,x,kind):
    changed=x.clone();cut=model.c.prefix+2;changed[:,cut:]=(changed[:,cut:]+1)%model.c.vocab
    original=model.hidden(x,ORDERS[0]);canonical=hidden(model,x,'block',PATHS[0])
    parity=float((original-canonical).abs().max());causal=[];batch=[]
    if kind=='deep':
        f=deep_features(model,x);g=deep_features(model,changed)
        causal.append(float((f[:,:cut-model.c.prefix+1]-g[:,:cut-model.c.prefix+1]).abs().max()))
        batch.append(float((f[:1]-deep_features(model,x[:1])).abs().max()))
    else:
        for path in PATHS:
            h=hidden(model,x,kind,path);z=hidden(model,changed,kind,path)
            causal.append(float((h[:,:cut]-z[:,:cut]).abs().max()))
            c=costs(model,x,kind,path);batch.append(float((c[:1]-costs(model,x[:1],kind,path)).abs().max()))
    if parity>2e-5 or max(causal)>2e-5 or max(batch)>2e-4:raise ValueError('Production numerical/causal gate failed '+repr((parity,causal,batch)))
    return dict(canonical_max=parity,causal_max=max(causal),batch_max=max(batch),precision='fp32_no_tf32')

@torch.no_grad()
def feature_cache(model,rows,ids,out,device,batch):
    out=Path(out);out.mkdir();f=np.lib.format.open_memmap(out/'features.npy',mode='w+',dtype=np.float32,shape=(len(ids),model.c.length-model.c.prefix,model.c.d))
    for i in range(0,len(ids),batch):
        x=rows.batch(ids[i:i+batch],device);f[i:i+len(x)]=deep_features(model,x).cpu().numpy()
        write_json(out/'progress.json',dict(rows=i+len(x),total=len(ids)))
    f.flush()
    if not np.isfinite(f).all():raise ValueError('Nonfinite features')
    return f


def check_inputs(config):
    for path,want in config['hashes'].items():
        if sha(path)!=want:raise ValueError('Changed input '+path)


def run(config,out,device='cuda'):
    out=Path(out)
    if out.exists():raise ValueError('Fresh output required')
    precision();torch.set_num_threads(4);start=time.monotonic();check_inputs(config)
    out.mkdir(parents=True);write_json(out/'config.json',config)
    state=torch.load(config['weights'],map_location='cpu',weights_only=False)
    model=Model(Config(**state['config'])).to(device);model.load_state_dict(state['model']);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    ids=np.load(config['indices']);kind=config['kind'];batch=config.get('batch',8)
    fitrows=Rows(config['stems']['fit'],model.c.length)
    write_json(out/'probe.json',probe(model,fitrows.batch(ids['fit'][:2],device),kind))
    head=None;head_seconds=0.;forward_count=0
    if kind=='deep':
        prior=Path(config['prior']);fs=[];cs=[]
        sample=fitrows.batch(ids['fit'][:2],device)
        cached=np.load(prior/'fit/costs.npy',mmap_mode='r')[:2]
        actual=torch.stack([token_cost(model,sample,p) for p in ORDERS],-1).cpu().numpy()
        error=float(np.max(np.abs(actual-cached)))
        if error>2e-4:raise ValueError('Cached costs/target/model mismatch '+str(error))
        write_json(out/'cache-probe.json',dict(max_error=error,gate=2e-4,rows=2,paths=8))
        for split in ('fit','valid'):
            fs.append(feature_cache(model,fitrows,ids[split],out/split,device,batch))
            cs.append(np.load(prior/split/'costs.npy',mmap_mode='r'));forward_count+=len(ids[split])
        t=time.monotonic();head=fit_head(fs[0],cs[0],fs[1],cs[1],55200+config['seed'],out/'head',device,epochs=config.get('epochs',8),batch=config.get('head_batch',4096));head_seconds=time.monotonic()-t
        del fs,cs
    for domain in ('dclm','fwe'):
        folder=out/domain;folder.mkdir();rows=Rows(config['stems'][domain],model.c.length);n=len(ids[domain]);targets=model.c.length-model.c.prefix
        if kind=='deep':
            old=np.load(prior/domain/'summary.npz');c=np.load(prior/domain/'costs.npy',mmap_mode='r')
            np.testing.assert_array_equal(old['ids'],ids[domain]);np.testing.assert_array_equal(old['clusters'],cluster_ids(rows,ids[domain]))
            choices=np.lib.format.open_memmap(folder/'choices.npy',mode='w+',dtype=np.uint8,shape=(n,targets))
            summary={k:old[k] for k in old.files};summary['stem']=summary.pop('learned');summary['stem_counts']=summary.pop('learned_counts');summary['stem_switches']=summary.pop('learned_switches')
            for i in range(0,n,batch):
                x=rows.batch(ids[domain][i:i+batch],device)
                with torch.no_grad():choices[i:i+len(x)]=head(deep_features(model,x)).argmax(-1).cpu().numpy()
                write_json(folder/'progress.json',dict(rows=i+len(x),total=n))
            choices.flush();summary['deep']=np.take_along_axis(c,choices[...,None],axis=2)[...,0].mean(1,dtype=np.float64)
            summary['deep_counts']=np.stack([(choices==a).sum(1) for a in range(8)],1);summary['deep_switches']=(choices[:,1:]!=choices[:,:-1]).sum(1)
            forward_count+=n
        else:
            c=np.lib.format.open_memmap(folder/'costs.npy',mode='w+',dtype=np.float32,shape=(n,targets,4))
            for i in range(0,n,batch):
                x=rows.batch(ids[domain][i:i+batch],device);c[i:i+len(x)]=torch.stack([costs(model,x,kind,path) for path in PATHS],-1).cpu().numpy()
                write_json(folder/'progress.json',dict(rows=i+len(x),total=n))
            c.flush();summary={f'path{j}':c[:,:,j].mean(1,dtype=np.float64) for j in range(4)}
            prior_dense=np.load(Path(config['prior'])/domain/'summary.npz')['canonical']
            np.testing.assert_allclose(summary['path0'],prior_dense,rtol=0,atol=2e-4)
            forward_count+=4*n
        summary.update(ids=ids[domain],clusters=cluster_ids(rows,ids[domain]))
        if any(not np.isfinite(v).all() for v in summary.values()):raise ValueError('Nonfinite summary')
        np.savez_compressed(folder/'summary.npz',**summary)
    check_inputs(config)
    write_json(out/'complete.json',dict(status='complete',name=config['name'],kind=kind,seconds=time.monotonic()-start,head_seconds=head_seconds,main_sequence_forwards=forward_count,
        probe_note='Additional parity probe: deep 25 full sequence paths including16 old-cache parity paths; component 32 full sequence paths. CPU/hash time included.',
        files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='progress.json'}))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.out,a.device)
if __name__=='__main__':main()
