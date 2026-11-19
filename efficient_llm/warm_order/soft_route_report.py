"""Replay terminal soft-mixture and hard-projection comparisons without test selection."""
import csv,json,shutil,tarfile
from pathlib import Path
import numpy as np
from .soft_route_prepare import NAMES
from .recursion_eval import check_train
from .strength_refinement import check
from .evaluate import paired_ci
from .train import write_json
from .data import sha

FLAGS=['reused_final_exploratory','two_seeds','multiple_comparisons','conditional_suffix480',
       'C6_GS18_candidate_blocks_not_equal_compute','soft_mixture_not_discrete_order','hard_projection_all_candidates_not_sparse',
       'fresh_optimizer_after_parent','weights_node_only']

def calculate(arrays):
    rows=[];ci={};mechanism=[];quality=[];discrete=[]
    for name in NAMES:
        for d in ('dclm','fwe'):
            a=arrays[name,d]
            for mode in ('soft','hard'):rows.append(dict(name=name,domain=d,metric=mode,nll=float(a[mode].mean(dtype=np.float64)),sequences=len(a[mode])))
            ci[f'{name}/{d}/hard-minus-soft']=paired_ci(a['hard']-a['soft'],a['clusters'])
    for s in (0,1):
        for d in ('dclm','fwe'):
            a=arrays[f'S-s{s}',d]
            for arm in ('C','G'):
                b=arrays[f'{arm}-s{s}',d]
                np.testing.assert_array_equal(a['ids'],b['ids']);np.testing.assert_array_equal(a['clusters'],b['clusters'])
                for mode in ('soft','hard'):ci[f'S-s{s}/{d}/{mode}-minus-{arm}']=paired_ci(a[mode]-b[mode],a['clusters'])
            mechanism.append(ci[f'S-s{s}/{d}/soft-minus-G']['mean']<=-.005)
            quality.append(mechanism[-1] and ci[f'S-s{s}/{d}/soft-minus-C']['mean']<=-.005)
            discrete.append(ci[f'S-s{s}/{d}/hard-minus-G']['mean']<=-.005 and ci[f'S-s{s}/{d}/hard-minus-C']['mean']<=-.005)
    return dict(rows=rows,comparisons=ci,soft_mechanism_all=all(mechanism),soft_quality_all=all(quality),discrete_quality_all=all(discrete),flags=FLAGS)

def check_arrays(a,length):
    n=len(a['ids'])
    if len(np.unique(a['ids']))!=n or any(not np.isfinite(v).all() for v in a.values()):raise ValueError('Invalid summaries')
    for mode in ('soft','hard'):
        counts=a[mode+'_counts'];sums=a[mode+'_probability_sums']
        if counts.shape!=(n,4,4) or sums.shape!=(n,4,4) or a[mode].shape!=(n,):raise ValueError('Summary shapes')
        if not np.issubdtype(counts.dtype,np.integer) or (counts<0).any() or (counts.sum(-1)!=length).any():raise ValueError('Invalid hard counts')
        if (sums<0).any():raise ValueError('Negative soft gate sum')
        np.testing.assert_allclose(sums.sum(-1),length,atol=.002,rtol=0)

def replay_run(out):
    out=Path(out);c=json.loads((out/'config.json').read_text());done=json.loads((out/'complete.json').read_text())
    check(c['hashes']);check_train(out/'train')
    if done['status']!='complete' or done['proof']['causal_max']!=0 or not done['proof']['gate_causality']:raise ValueError('Incomplete or noncausal run')
    for p,h in done['files'].items():
        if sha(out/p)!=h:raise ValueError('Changed result '+p)
    for d in ('dclm','fwe'):check_arrays(dict(np.load(out/d/'summary.npz')),c['train']['model']['length'])
    return done

def report(base,backup):
    base=Path(base);backup=Path(backup);arrays={};done={};files=[]
    for name in NAMES:
        run=base/'runs'/name;done[name]=replay_run(run)
        for d in ('dclm','fwe'):arrays[name,d]=dict(np.load(run/d/'summary.npz'))
        files += [p for p in run.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.npz')]
    result=calculate(arrays);result.update(runs=done,node_replay='PASS');write_json(base/'REPORT.json',result)
    with (base/'RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(result['rows'][0]));w.writeheader();w.writerows(result['rows'])
    lines=['# Continuous block mixtures and hard projection','','| Run | Domain | Mode | NLL |','|---|---|---|---:|']
    lines += [f"|{r['name']}|{r['domain']}|{r['metric']}|{r['nll']:.6f}|" for r in result['rows']]
    lines += ['',f"Soft mechanism: {result['soft_mechanism_all']}; soft quality: {result['soft_quality_all']}; discrete quality: {result['discrete_quality_all']}",'',', '.join(FLAGS)]
    (base/'REPORT.md').write_text('\n'.join(lines)+'\n')
    files += [base/p for p in ('REPORT.json','REPORT.md','RESULTS.csv','inputs/complete.json','execution/source.json','execution/admission.json')]
    members={str(p.relative_to(base)):sha(p) for p in files};write_json(base/'export-members.json',members);files.append(base/'export-members.json')
    archive=base/'results.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(base)),recursive=False)
    if archive.stat().st_size>=25_000_000:raise ValueError('Export too large')
    backup.mkdir(parents=True,exist_ok=False);copies={}
    for p in [archive,*sorted((base/'runs').glob('*/train/*.pt'))]:
        target=backup/p.relative_to(base);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        if sha(target)!=sha(p):raise ValueError('Backup mismatch')
        copies[str(target)]=sha(target)
    if len(copies)!=13:raise ValueError('Backup inventory mismatch')
    write_json(base/'backup.json',dict(archive_sha256=sha(archive),bytes=archive.stat().st_size,files=members,node_copies=copies,durability='Same-node copies; weights and full state node-only'))
    return result
