"""Authenticate recovered dense parents and identical six-arm router rows."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .data import sha, Rows
from .strength_refinement import check
from .pair_order import load_initial
from .pair_order_report import NAMES
from .train import write_json


def prepare(plan,out):
    out=Path(out)
    if out.exists():raise ValueError('Fresh inputs required')
    check(plan['hashes']);t=plan['training'];rows=Rows(t['data'],t['model']['length'])
    need=t['steps']*t['batch'];orders={}
    for seed in (0,1):
        p=plan['orders'][str(seed)];ids=np.load(p,allow_pickle=False)
        if (plan['hashes'].get(p)!=sha(p) or ids.ndim!=1 or ids.dtype.kind not in 'iu' or
                len(ids)!=need or len(np.unique(ids))!=need or ids.min()<0 or ids.max()>=len(rows.ids)):
            raise ValueError('Invalid common router rows')
        if rows.starts[ids,1:].any():raise ValueError('Cross-document fit row')
        orders[seed]=ids
    if np.intersect1d(orders[0],orders[1]).size:raise ValueError('Seed rows overlap')
    out.mkdir(parents=True)
    for name in NAMES:
        arm,seed=name.split('-s');seed=int(seed);parent=plan['parents'][str(seed)]
        for f,h in parent['hashes'].items():
            if plan['hashes'].get(str(Path(parent['directory'])/f))!=h:raise ValueError('Unpinned parent')
        c=dict(t,name=name,arm=arm,seed=seed,parent=parent['directory'],parent_hashes=parent['hashes'],
               parent_steps=plan['parent_steps'],indices=plan['orders'][str(seed)])
        c['expected_inputs']={p:sha(p) for p in (c['data']+'.bin',c['data']+'.starts',c['indices'])}
        torch.manual_seed(seed);m=load_initial(c);del m
        write_json(out/f'{name}.json',dict(name=name,train=c,eval=plan['eval'],hashes=plan['hashes']))
        print('PAIR_ORDER_PARENT_PASS',name,flush=True)
    for name,control in plan['controls'].items():
        done=json.loads(Path(control['complete']).read_text())
        if (name!=f"P-s{done['seed']}" or done['status']!='complete' or done['train']['weights_sha256']!=plan['parents'][str(done['seed'])]['hashes']['weights.pt']):
            raise ValueError('Wrong dense control')
        for d,p in control['arrays'].items():
            if sha(p)!=plan['hashes'][p] or done['files'][d+'/summary.npz']!=sha(p):raise ValueError('Control array identity')
    write_json(out/'complete.json',dict(status='ready',hashes=plan['hashes'],controls=plan['controls'],length=t['model']['length'],
        rows_reused_from_body=True,seed_overlap=0,files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print('PAIR_ORDER_INPUTS_READY',sha(out/'complete.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    prepare(json.loads(Path(a.plan).read_text()),a.out)
