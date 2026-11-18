"""Verify frozen parents and document-separated cost-head reservations."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .data import Rows,sha
from .evaluate import cluster_ids
from .train import write_json
from .strength_refinement import check
from .causal_cost_report import NAMES


def prepare(plan,out):
    out=Path(out)
    if out.exists():raise ValueError('Fresh preparation required')
    check(plan['hashes'])
    if not json.loads(Path(plan['parent_report']).read_text())['LF_both_seeds_headroom_gt_005']['hold']:raise ValueError('Registered diagnostic trigger absent')
    ids=np.load(plan['indices']);fit=ids['fit'][:plan['fit_n']];valid=ids['valid'][plan['valid_start']:plan['valid_stop']]
    if len(fit)!=plan['fit_n'] or len(valid)!=plan['valid_stop']-plan['valid_start'] or len(valid)<4:raise ValueError('Reservation size')
    rows=Rows(plan['stems']['fit'],plan['length'])
    if set(cluster_ids(rows,fit)) & set(cluster_ids(rows,ids['valid'])):raise ValueError('Fit/validation document overlap')
    if len(np.unique(fit))!=len(fit) or len(np.unique(valid))!=len(valid):raise ValueError('Repeated row')
    # Final corpora have different stems; verify no row-address aliasing.
    if plan['stems']['fit'] in (plan['stems']['dclm'],plan['stems']['fwe']):raise ValueError('Final/train stem alias')
    out.mkdir(parents=True)
    for name in NAMES:
        arm,seed=name.split('-');seed=int(seed[1:]);weights=plan['weights'][arm][str(seed)];state=torch.load(weights,map_location='cpu',weights_only=False)
        if state['arm']!=arm or state['seed']!=seed or state['config']['length']!=plan['length']:raise ValueError('Parent metadata')
        c=dict(name=name,arm=arm,seed=seed,weights=weights,dense_weights=plan['weights']['D6C'][str(seed)],stems=plan['stems'],indices=plan['indices'],fit_n=plan['fit_n'],valid_start=plan['valid_start'],valid_stop=plan['valid_stop'],length=plan['length'],window=state['window'],stride=state['stride'],hashes=plan['hashes'],epochs=plan.get('epochs',30),hidden=plan.get('hidden',128))
        write_json(out/(name+'.json'),c)
    write_json(out/'complete.json',dict(status='ready',hashes=plan['hashes'],fit_validation_document_overlap=0,files={p.name:sha(p) for p in out.iterdir() if p.is_file()}));print('CAUSAL_COST_INPUTS_READY',sha(out/'complete.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True);a=p.parse_args();prepare(json.loads(Path(a.plan).read_text()),a.out)
