"""Registered full continuation/study entry points; gate and ownership are separate.

Called only by an admitted owned queue. No remote operations or automatic launch.
"""
import argparse,json
from pathlib import Path

from .adaptation_train import train,check_runtime
from .adaptation_reservation import validate_binding
from .adaptation_study import run as study
from .data import sha
from .horizon_train import write_json


def bound_config(prepared, prepared_sha256, name, gate, gate_sha256):
    prepared=Path(prepared)
    if name not in ('A-s0','B-s0','A-s1','B-s1'):raise ValueError('Registered paired branch required')
    if sha(prepared/'manifest.json')!=prepared_sha256:raise ValueError('Prepared manifest changed')
    manifest=json.loads((prepared/'manifest.json').read_text());path=prepared/(name+'.json')
    if sha(path)!=manifest['files'][name+'.json']:raise ValueError('Prepared config changed')
    c=json.loads(path.read_text());validate_binding(c)
    c['gpu_admission']=dict(path=str(gate),sha256=gate_sha256)
    check_runtime(c,'cuda',None,False)
    return c


def continue_branch(prepared, prepared_sha256, name, gate, gate_sha256, out):
    c=bound_config(prepared,prepared_sha256,name,gate,gate_sha256)
    return train(c,out,device='cuda')


def measure_seed(base,seed,out):
    base=Path(base)
    if seed not in (0,1):raise ValueError('Registered seed required')
    recipes={b:json.loads((base/'runs'/f'{b}-s{seed}'/'recipe.json').read_text()) for b in ('A','B')}
    a=recipes['A']['config'];b=recipes['B']['config']
    if {k:v for k,v in a.items() if k!='branch'}!={k:v for k,v in b.items() if k!='branch'}:raise ValueError('Unmatched branch configs')
    for branch,c in [('A',a),('B',b)]:
        check_runtime(c,'cuda',None,False)
        directory=base/'runs'/f'{branch}-s{seed}';done=json.loads((directory/'complete.json').read_text())
        if done['step']!=c['steps'] or sha(directory/'fullstate.pt')!=done['fullstate_sha256']:raise ValueError('Training incomplete/changed')
    snapshots=[]
    for step in a['snapshots']:
        phase={}
        for branch in ('A','B'):
            p=base/'runs'/f'{branch}-s{seed}'/'snapshots'/f'{step:06d}'/'fullstate.pt'
            cp=json.loads(p.with_name('checkpoint.json').read_text())
            if cp['step']!=step or sha(p)!=cp['sha256']:raise ValueError('Snapshot bytes/step changed')
            phase[branch]=dict(path=str(p),sha256=cp['sha256'])
        snapshots.append(dict(step=step,phase_files=phase))
    # These six keys and four positions are fixed before GPU study outcomes.
    keys=['slot/2/input','slot/3/input','block/2/attention_update','block/2/ffn_update',
          'block/3/attention_update','block/3/ffn_update']
    config=dict(schema=1,domain='dclm',parent=a['parent'],parent_hashes=a['parent_hashes'],
                snapshots=snapshots,data=a['data'],indices=a['confirmation_indices'],batch_size=1,
                expected_inputs={p:a['expected_inputs'][p] for p in (a['data']+'.bin',a['data']+'.starts',a['confirmation_indices'])},
                positions=[31,127,255,511],geometry_keys=keys,max_moment_bytes=128*1024**2,
                measurement_device='cuda',
                flags=['predeclared_contrasts','exploratory','historical_overlap_unknown','no_document_freshness_claim'])
    return study(config,out)


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    t=sub.add_parser('train');t.add_argument('--prepared',required=True);t.add_argument('--prepared-sha256',required=True)
    t.add_argument('--name',required=True);t.add_argument('--gate',required=True);t.add_argument('--gate-sha256',required=True);t.add_argument('--out',required=True)
    e=sub.add_parser('study');e.add_argument('--base',required=True);e.add_argument('--seed',type=int,required=True);e.add_argument('--out',required=True)
    a=p.parse_args()
    if a.mode=='train':continue_branch(a.prepared,a.prepared_sha256,a.name,a.gate,a.gate_sha256,a.out)
    else:measure_seed(a.base,a.seed,a.out)

if __name__=='__main__':main()
