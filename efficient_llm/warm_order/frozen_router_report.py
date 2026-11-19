"""Replay frozen-body router screens against immutable dense parents."""
import csv
import json
import shutil
import tarfile
from pathlib import Path
import numpy as np
from .data import sha
from .evaluate import paired_ci
from .exposure_recovery_report import check_arrays
from .strength_refinement import check
from .recursion_eval import check_train
from .train import write_json

NAMES=[f'{a}-s{s}' for a in ('G','L','M') for s in (0,1)]
ALL=NAMES+[f'C-s{s}' for s in (0,1)]
FLAGS=['reused_final_exploratory','multiple_comparisons','two_seeds','repeated_body_training_rows',
       'conditional_suffix480','unequal_compute_C6_routed18','soft_not_discrete_order',
       'hard_projection_not_sparse_speed','weights_node_only','router_capacity_package_not_parameter_isolation']


def calculate(arrays):
    rows=[];comparisons={};soft=[];hard=[];dynamic=[]
    for name in ALL:
        for domain in ('dclm','fwe'):
            a=arrays[name,domain]
            for mode in ('soft','hard'):
                rows.append(dict(name=name,domain=domain,metric=mode,nll=float(a[mode].mean(dtype=np.float64)),sequences=len(a[mode])))
            comparisons[f'{name}/{domain}/hard-minus-soft']=paired_ci(a['hard']-a['soft'],a['clusters'])
    for seed in (0,1):
        for domain in ('dclm','fwe'):
            a=arrays[f'M-s{seed}',domain]
            for arm in ('C','G','L'):
                b=arrays[f'{arm}-s{seed}',domain]
                np.testing.assert_array_equal(a['ids'],b['ids']);np.testing.assert_array_equal(a['clusters'],b['clusters'])
                for mode in ('soft','hard'):
                    comparisons[f'M-s{seed}/{domain}/{mode}-minus-{arm}']=paired_ci(a[mode]-b[mode],a['clusters'])
            soft.append(all(comparisons[f'M-s{seed}/{domain}/soft-minus-{arm}']['mean']<=(-.005 if arm=='C' else -.001) for arm in ('C','G','L')))
            hard.append(all(comparisons[f'M-s{seed}/{domain}/hard-minus-{arm}']['mean']<=(-.005 if arm=='C' else -.001) for arm in ('C','G','L')))
            counts=a['hard_counts'];totals=counts.sum(0)
            dynamic.append(bool(totals.sum()>np.trace(totals) and len(np.unique(counts.reshape(len(counts),-1),axis=0))>1))
    return dict(rows=rows,comparisons=comparisons,soft_quality_all=all(soft),discrete_quality_all=all(hard),
                noncanonical_input_dependent_all=all(dynamic),dynamic_order_screen_all=all(hard) and all(dynamic),flags=FLAGS)


def replay_run(run):
    run=Path(run);c=json.loads((run/'config.json').read_text());done=json.loads((run/'complete.json').read_text())
    check(c['hashes']);check_train(run/'train');p=done['proof']
    if (done['status']!='complete' or p['causal_max']!=0 or not p['gate_causality'] or
            not p['initial']['hard_parent_exact'] or not p['frozen_body_exact'] or
            p['relative_gradient_error']>.08 or p['loss_delta']>.03 or
            p['resume']['loaded_fullstate_sha256']!=p['first_checkpoint']['sha256']):
        raise ValueError('Run gate failure')
    if any(max(p[d]['fullset_nll_parity'].values())>2e-4 for d in ('dclm','fwe')):raise ValueError('Reference parity failure')
    for f,h in done['files'].items():
        if sha(run/f)!=h:raise ValueError('Result hash mismatch')
    for d in ('dclm','fwe'):check_arrays(dict(np.load(run/d/'summary.npz')),c['train']['model']['length'])
    return done


def report(base,backup):
    base=Path(base);backup=Path(backup);prepared=json.loads((base/'inputs/complete.json').read_text())
    check(prepared['hashes']);arrays={};runs={};files=[]
    for name in NAMES:
        run=base/'runs'/name;runs[name]=replay_run(run)
        for d in ('dclm','fwe'):arrays[name,d]=dict(np.load(run/d/'summary.npz'))
        files.extend(p for p in run.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.npz'))
    for name,control in prepared['controls'].items():
        for d,p in control['arrays'].items():
            if sha(p)!=prepared['hashes'][p]:raise ValueError('Changed dense control')
            dest=base/'controls'/name/d/'summary.npz';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
            arrays[name,d]=dict(np.load(dest));check_arrays(arrays[name,d],prepared['length']);files.append(dest)
    result=calculate(arrays);result.update(runs=runs,node_replay='PASS');write_json(base/'REPORT.json',result)
    with (base/'RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(result['rows'][0]));w.writeheader();w.writerows(result['rows'])
    lines=['# Frozen-backbone router capacity','','| Run | Domain | Mode | NLL |','|---|---|---|---:|']
    lines += [f"|{r['name']}|{r['domain']}|{r['metric']}|{r['nll']:.6f}|" for r in result['rows']]
    lines += ['',f"Soft quality: {result['soft_quality_all']}; discrete quality: {result['discrete_quality_all']}",'',', '.join(FLAGS)]
    (base/'REPORT.md').write_text('\n'.join(lines)+'\n')
    files += [base/p for p in ('REPORT.json','REPORT.md','RESULTS.csv','inputs/complete.json','execution/source.json','execution/admission.json')]
    members={str(p.relative_to(base)):sha(p) for p in files};write_json(base/'export-members.json',members);files.append(base/'export-members.json')
    archive=base/'results.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(base)),recursive=False)
    if archive.stat().st_size>=25_000_000:raise ValueError('Oversized results')
    backup.mkdir(parents=True,exist_ok=False);copies={}
    for p in [archive,*sorted((base/'runs').glob('*/train/*.pt'))]:
        dest=backup/p.relative_to(base);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        if sha(dest)!=sha(p):raise ValueError('Backup mismatch')
        copies[str(dest)]=sha(dest)
    if len(copies)!=13:raise ValueError('Backup inventory mismatch')
    write_json(base/'backup.json',dict(archive_sha256=sha(archive),bytes=archive.stat().st_size,files=members,node_copies=copies,durability='Same-node only'))
    return result
