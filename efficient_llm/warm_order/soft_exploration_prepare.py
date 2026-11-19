"""Reuse authenticated matched rows/parents and six completed controls."""
import json
from pathlib import Path
from .data import sha
from .train import write_json
from .strength_refinement import check
from .soft_exploration import load_initial

NEW_NAMES=[f'{a}F-s{s}' for a in ('G','S') for s in (0,1)]
CONTROL_NAMES=[f'{a}-s{s}' for a in ('C','G','S') for s in (0,1)]
SCHEDULE='floor0.4-ramp500-hold500-decay500-zero500'


def prepare(plan,out):
    out=Path(out)
    if out.exists():raise ValueError('Fresh preparation required')
    check(plan['hashes'])
    prior=json.loads(Path(plan['control_report']).read_text())
    if prior['node_replay']!='PASS':raise ValueError('Unverified controls')
    out.mkdir(parents=True)
    for name in NEW_NAMES:
        arm=name[0];seed=int(name[-1]);old=Path(plan['controls'][f'{arm}-s{seed}'])
        c=json.loads((old/'config.json').read_text());t=c['train']
        if t['arm']!=arm or t['seed']!=seed:raise ValueError('Control identity mismatch')
        load_initial(t)
        t['exploration_schedule']=SCHEDULE
        c.update(name=name,hashes={**c['hashes'],**plan['hashes']})
        write_json(out/f'{name}.json',c)
        print('EXPLORATION_PARENT_ROWS_PASS',name,sha(t['indices']),flush=True)
    write_json(out/'complete.json',dict(hashes=plan['hashes'],controls=plan['controls'],files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print('EXPLORATION_INPUTS_READY',sha(out/'complete.json'),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True);a=p.parse_args();prepare(json.loads(Path(a.plan).read_text()),a.out)
