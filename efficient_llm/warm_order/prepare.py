"""CPU-only immutable row selections from restored, golden-verified data."""
import argparse,json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .data import Rows,sha
from .model import Config
from .train import write_json

def prepare(plan,out):
    out=Path(out)
    if out.exists():raise ValueError('Preparation already exists; retrieve receipt')
    for p,digest in plan['files'].items():
        if sha(p)!=digest:raise ValueError('Golden data mismatch '+p)
    pools={k:Rows(v).eligible() for k,v in plan['stems'].items()}
    out.mkdir(parents=True)
    for seed in (0,1):
        rng=np.random.default_rng(8800+seed);train_ids=rng.permutation(pools['fit'])
        need=12000*32
        if len(train_ids)<need+8192:raise ValueError('Too few eligible training windows')
        np.save(out/f'train-s{seed}.npy',train_ids[:need])
        dev=np.random.default_rng(9900).permutation(pools['dev'])[:2048]
        test=np.random.default_rng(9901).permutation(pools['test'])[:8192]
        if len(dev)<2048 or len(test)<8192:raise ValueError('Insufficient prespecified held-out windows')
        np.savez(out/f'eval-s{seed}.npz',fit=train_ids[need:need+8192],dev=dev,test=test)
        for arm in ('D','R','W'):
            indices=str(out/f'train-s{seed}.npy');data=plan['stems']['fit']
            expected={p:plan['files'][p] for p in (data+'.bin',data+'.starts')};expected[indices]=sha(indices)
            config=dict(data=data,indices=indices,expected_inputs=expected,batch=32,steps=12000,arm=arm,seed=seed,lr=.0006,warmup=300,threads=4,save_every=1000,log_every=50,model=asdict(Config()))
            write_json(out/f'{arm}-s{seed}.json',config)
    write_json(out/'complete.json',dict(status='ready',input_hashes=plan['files'],eligible_counts={k:len(v) for k,v in pools.items()},files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print((out/'complete.json').read_text(),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True);a=p.parse_args();prepare(json.loads(Path(a.plan).read_text()),a.out)
if __name__=='__main__':main()
