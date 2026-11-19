"""Authenticate recovered dense parents and identical six-arm router rows."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .data import sha, Rows
from .strength_refinement import check
from .pair_order import load_initial
from .restart_factorial_report import NAMES
from .restart_factorial import initialize_optimizer
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
        c=dict(t,name=name,arm='C',factor=arm,seed=seed,parent=parent['directory'],parent_hashes=parent['hashes'],
               parent_steps=plan['parent_steps'],indices=plan['orders'][str(seed)])
        c['expected_inputs']={p:sha(p) for p in (c['data']+'.bin',c['data']+'.starts',c['indices'])}
        torch.manual_seed(seed);m=load_initial(c);o,proof=initialize_optimizer(m,c,'cpu');del m,o
        write_json(out/f'{name}.json',dict(name=name,train=c,eval=plan['eval'],hashes=plan['hashes']))
        print('RESTART_FACTORIAL_PARENT_PASS',name,flush=True)
    if set(plan['controls'])!={f'{a}-s{s}' for a in ('P','FH') for s in (0,1)}:
        raise ValueError('Missing factorial references')
    for name,control in plan['controls'].items():
        done=json.loads(Path(control['complete']).read_text())
        if done['status']!='complete' or done['seed']!=int(name[-1]):raise ValueError('Wrong control identity')
        if name.startswith('P-'):
            if done['train']['weights_sha256']!=plan['parents'][name[-1]]['hashes']['weights.pt']:raise ValueError('Wrong parent control')
        else:
            rc=json.loads(Path(control['recipe']).read_text());old=rc['config'];new=json.loads((out/f'FL-s{name[-1]}.json').read_text())['train']
            same=('parent','parent_hashes','parent_steps','model','seed','data','indices','expected_inputs','steps','batch','lr','warmup','canonical_probability','router_width','threads')
            if old['arm']!='C' or old['schedule']!='paired-body-router-warmup-cosine' or any(old[k]!=new[k] for k in same):
                raise ValueError('FH recipe does not match factorial')
            if rc['torch']!=torch.__version__ or rc['numpy']!=np.__version__:raise ValueError('FH runtime mismatch')
            source_root=Path(__file__).resolve().parents[2]
            if any(sha(source_root/path)!=digest for path,digest in rc['source'].items()):raise ValueError('Historical FH source changed')
            if done['train']['fullstate_sha256']!=control['fullstate_sha256']:raise ValueError('FH checkpoint identity')
        for d,p in control['arrays'].items():
            if sha(p)!=plan['hashes'][p] or done['files'][d+'/summary.npz']!=sha(p):raise ValueError('Control array identity')
    write_json(out/'complete.json',dict(status='ready',hashes=plan['hashes'],controls=plan['controls'],length=t['model']['length'],
        rows_reused_from_body=True,seed_overlap=0,files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print('RESTART_FACTORIAL_INPUTS_READY',sha(out/'complete.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    prepare(json.loads(Path(a.plan).read_text()),a.out)
