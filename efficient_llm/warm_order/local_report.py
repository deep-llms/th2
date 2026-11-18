"""Independent array replay, paired comparisons and node-only checkpoint backup."""
import json,shutil,tarfile
from pathlib import Path
import numpy as np
from .data import sha
from .train import write_json
from .evaluate import paired_ci

NAMES=[f'{arm}-s{seed}' for arm in ('C','U','CF','PG') for seed in (0,1)]


def check_train(run,steps=4000):
    done=json.loads((run/'complete.json').read_text())
    if done['step']!=steps:raise ValueError('Nonterminal training')
    for f,k in [('weights.pt','weights_sha256'),('fullstate.pt','fullstate_sha256')]:
        if sha(run/f)!=done[k]:raise ValueError('Checkpoint changed '+str(run/f))
    return done


def report(base,backup,steps=4000):
    base=Path(base);backup=Path(backup);reports=[];arrays={};train=[]
    for name in NAMES:
        run=base/'runs'/name;train.append(check_train(run,steps));ev=base/'eval'/name
        row=json.loads((ev/'report.json').read_text())
        if row['weights_sha256']!=train[-1]['weights_sha256']:raise ValueError('Evaluated wrong checkpoint')
        for rel,want in json.loads((ev/'complete.json').read_text())['files'].items():
            if sha(ev/rel)!=want:raise ValueError('Evaluation changed')
        constant=0
        if row['arm']!='C':
            fit=np.load(ev/'fit_costs.npz',allow_pickle=False)['costs'];constant=int(fit.mean(0).argmin())
            if constant!=row['constant']:raise ValueError('Constant route selection mismatch')
        for domain in ('dclm','fwe'):
            a=np.load(ev/f'{domain}_costs.npz',allow_pickle=False);arrays[name,domain]=a
            cost=a['costs'];choices=a['choices'];metrics={'canonical':cost[:,0]}
            if row['arm']!='C':
                metrics.update(uniform=cost.mean(1),constant=cost[:,constant],oracle=cost.min(1),mixture=cost@np.array([.79,.07,.07,.07],dtype=np.float32))
            if row['arm'] in ('CF','PG'):metrics['learned']=cost[np.arange(len(cost)),choices]
            if set(metrics)!=set(row['domains'][domain]['metrics']):raise ValueError('Metric coverage mismatch')
            for k,v in metrics.items():
                np.testing.assert_allclose(v,a[k],atol=1e-7,rtol=0)
                if abs(float(v.mean())-row['domains'][domain]['metrics'][k])>1e-7:raise ValueError('NLL mismatch')
        reports.append(row)
    comparisons={}
    for domain in ('dclm','fwe'):
        for seed in (0,1):
            dense=arrays[f'C-s{seed}',domain];uniform=arrays[f'U-s{seed}',domain]
            for arm in ('U','CF','PG'):
                a=arrays[f'{arm}-s{seed}',domain]
                np.testing.assert_array_equal(a['ids'],dense['ids']);np.testing.assert_array_equal(a['clusters'],dense['clusters'])
                key='mixture' if arm=='U' else 'learned'
                comparisons[f'{domain}_{arm}_minus_C_s{seed}']=paired_ci(a[key]-dense['canonical'],dense['clusters'])
                if arm!='U':comparisons[f'{domain}_{arm}_minus_U_s{seed}']=paired_ci(a[key]-uniform['mixture'],dense['clusters'])
    flags=json.loads((base/'inputs/complete.json').read_text())['flags']
    result=dict(rows=reports,training=train,comparisons=comparisons,flags=flags)
    write_json(base/'REPORT.json',result)
    lines=['# Dense-init local-order screening','','Conditional 32-prefix/480-suffix NLL; two seeds; extra forward compute differs.','','| Arm | Seed | DCLM NLL | FWE NLL |','|---|---:|---:|---:|']
    for r in reports:
        k='canonical' if r['arm']=='C' else 'mixture' if r['arm']=='U' else 'learned'
        lines.append(f"| {r['arm']} | {r['seed']} | {r['domains']['dclm']['metrics'][k]:.6f} | {r['domains']['fwe']['metrics'][k]:.6f} |")
    lines+=['','No equal-compute or general-LM superiority claim. Constant routes are selected only on separate training-domain windows. Oracle is diagnostic. See REPORT.json for routing counts and document-cluster intervals.']
    (base/'REPORT.md').write_text('\n'.join(lines)+'\n')
    files=[base/'REPORT.json',base/'REPORT.md',base/'inputs/complete.json',base/'execution/source.json']
    files+=list(base.glob('gpu-probe-*.json'))
    for folder in ('runs','eval'):
        files.extend(p for p in (base/folder).rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.npz'))
    if any(p.suffix=='.pt' for p in files):raise ValueError('Weights forbidden in small export')
    with tarfile.open(base/'results.tar.gz','w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(base)),recursive=False)
    if (base/'results.tar.gz').stat().st_size>25_000_000:raise ValueError('Small result cap exceeded')
    backup.mkdir(parents=True,exist_ok=False);copies={}
    for name in NAMES:
        (backup/name).mkdir()
        for f in ('weights.pt','fullstate.pt','recipe.json','complete.json','checkpoint.json'):
            src=base/'runs'/name/f;dst=backup/name/f;shutil.copyfile(src,dst)
            want=sha(src)
            if sha(dst)!=want:raise ValueError('Backup mismatch')
            copies[str(dst)]=want
    shutil.copyfile(base/'results.tar.gz',backup/'results.tar.gz')
    if sha(backup/'results.tar.gz')!=sha(base/'results.tar.gz'):raise ValueError('Result backup mismatch')
    write_json(base/'backup.json',dict(node_checkpoint_copies=copies,files={str(p.relative_to(base)):sha(p) for p in files},
               archive_sha256=sha(base/'results.tar.gz'),bytes=(base/'results.tar.gz').stat().st_size,weights_off_node=False))
    return result
