"""Seeded KG and document-aware fixed-order token streams; all local inputs."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from . import SPEC_SHA

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(dict(value,spec_sha256=SPEC_SHA),indent=2,allow_nan=False)+'\n');tmp.replace(p)

def make_kg(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(2026)
    relation=np.stack([rng.choice(200,20,replace=False) for _ in range(2000)])
    target=rng.integers(0,2000,(2000,20));edges=np.stack([np.repeat(np.arange(2000),20),relation.ravel()+2000,target.ravel()],-1).astype(np.int64)
    permutation=rng.permutation(40000);ood=np.zeros(40000,dtype=bool);ood[permutation[:2000]]=True
    ids=np.arange(40000);first=np.repeat(ids,20);second=(target.ravel()[first]*20+np.tile(np.arange(20),40000))
    two=np.stack([edges[first,0],edges[first,1],edges[second,1],edges[second,2]],-1)
    ii=np.flatnonzero(~ood[first]&~ood[second]);oo=np.flatnonzero(ood[first]&ood[second]);rng.shuffle(ii);rng.shuffle(oo)
    atomic=np.column_stack([edges,np.full(40000,2200)])
    train=np.concatenate([atomic,two[ii[2000:2000+273600]]]);assert len(train)==313600
    groups=dict(train=train,atoms=atomic,ood_atoms=atomic[ood],id_dev=two[ii[:1000]],id_test=two[ii[1000:2000]],ood_dev=two[oo[:len(oo)//2]],ood_test=two[oo[len(oo)//2:]])
    test_edges=second[oo[len(oo)//2:]];groups['ood_bridge']=atomic[test_edges]
    groups['ood_dev_bridge']=atomic[second[oo[:len(oo)//2]]]
    for name,arr in groups.items():np.save(root/(name+'.npy'),arr)
    np.savez(root/'kg.npz',atomic=edges,ood=ood,first=first,second=second)
    write(root/'manifest.json',dict(seed=2026,entities=2000,relations=200,atomic=40000,train=313600,files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size,rows=len(groups[p.stem]) if p.stem in groups else None) for p in root.iterdir() if p.is_file()}))
    return root

def e1_batch(rows,device):
    tokens=torch.as_tensor(np.array(rows),dtype=torch.long,device=device);segments=torch.where(tokens==2200,-1,0);labels=torch.full_like(tokens,-100)
    answer=torch.where(tokens[:,-1]==2200,2,3);labels[torch.arange(len(tokens),device=device),answer]=tokens[torch.arange(len(tokens),device=device),answer]
    positions=torch.arange(4,device=device).expand_as(tokens)
    return dict(tokens=tokens,segments=segments,positions=positions,labels=labels)

def e2_batch(rows,device,eos=0,document_starts=None):
    # Source admissions must prove appended EOS document boundaries before use.
    tokens=torch.as_tensor(np.array(rows),dtype=torch.long,device=device);starts=torch.zeros_like(tokens,dtype=torch.bool);starts[:,0]=True;starts[:,1:]=tokens[:,:-1]==eos
    if document_starts is not None:
        starts=torch.as_tensor(np.array(document_starts),dtype=torch.bool,device=device);starts[:,0]=True
    segments=starts.cumsum(1)-1;idx=torch.arange(tokens.shape[1],device=device).expand_as(tokens);positions=idx-torch.where(starts,idx,0).cummax(1).values
    labels=tokens.clone();labels[starts]=-100
    return dict(tokens=tokens,segments=segments,positions=positions,labels=labels)

class Stream:
    def __init__(self,root,name):
        self.root=Path(root);self.name=name
        self.ids=np.memmap(self.root/(name+'.bin'),mode='r',dtype='<u4').reshape(-1,512)
        self.starts=np.memmap(self.root/(name+'.starts'),mode='r',dtype='u1').reshape(-1,512)
        if self.ids.shape!=self.starts.shape:raise ValueError('Boundary shape differs')
    def __len__(self):return len(self.ids)
    def batch(self,indices,device):return e2_batch(self.ids[indices],device,document_starts=self.starts[indices])
