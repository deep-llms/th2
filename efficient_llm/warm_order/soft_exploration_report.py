"""Replay terminal soft-mixture and hard-projection comparisons without test selection."""
import csv,json,shutil,tarfile
from pathlib import Path
import numpy as np
from .soft_exploration_prepare import NEW_NAMES,CONTROL_NAMES
NAMES=CONTROL_NAMES+NEW_NAMES
from .recursion_eval import check_train
from .strength_refinement import check
from .evaluate import paired_ci
from .train import write_json
from .data import sha

FLAGS=['reused_final_exploratory','two_seeds','multiple_comparisons','conditional_suffix480',
       'C6_GS18_candidate_blocks_not_equal_compute','soft_mixture_not_discrete_order','hard_projection_all_candidates_not_sparse',
       'fresh_optimizer_after_parent','weights_node_only']

def calculate(arrays):
    from .soft_route_report import calculate as baseline_calculate
    selected={(name,d):arrays[(name if name.startswith('C-') else name[0]+'F'+name[1:]),d] for name in CONTROL_NAMES for d in ('dclm','fwe')}
    result=baseline_calculate(selected)
    result['rows']=[]
    for name in NAMES:
        for d in ('dclm','fwe'):
            a=arrays[name,d]
            for mode in ('soft','hard'):
                result['rows'].append(dict(name=name,domain=d,metric=mode,nll=float(a[mode].mean(dtype=np.float64)),sequences=len(a[mode])))
    result['comparisons']={k.replace('S-s','SF-s').replace('G-s','GF-s').replace('minus-G','minus-GF'):v for k,v in result['comparisons'].items()}
    dynamic=[]
    for name in NEW_NAMES:
        old=name.replace('F-','-')
        for d in ('dclm','fwe'):
            a,b=arrays[name,d],arrays[old,d]
            np.testing.assert_array_equal(a['ids'],b['ids']);np.testing.assert_array_equal(a['clusters'],b['clusters'])
            for mode in ('soft','hard'):result['comparisons'][f'{name}/{d}/{mode}-minus-{old}']=paired_ci(a[mode]-b[mode],a['clusters'])
            if name.startswith('S'):
                counts=a['hard_counts'].sum(0);off=counts.sum()-np.trace(counts)
                diverse=any(np.count_nonzero(row)>1 for row in counts)
                input_variable=len(np.unique(a['hard_counts'].reshape(len(a['ids']),-1),axis=0))>1
                dynamic.append(bool(off>0 and diverse and input_variable))
    result['noncanonical_diverse_hard_all']=all(dynamic)
    result['dynamic_order_screen_all']=result['discrete_quality_all'] and all(dynamic)
    result['flags']=FLAGS+['repeated_matched_rows_for_intervention','uniform_floor_preserves_argmax_at_fixed_logits','final_epsilon_zero']
    return result

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
    if c['train'].get('exploration_schedule'):
        if done['proof']['floor_causality']!={'causal_max':0.,'gate_causality':True}:raise ValueError('Floor causality proof missing')
        trace=[json.loads(x) for x in (out/'train/trace.jsonl').read_text().splitlines()]
        if c['train']['steps']==2000 and trace[-1]['exploration_epsilon']!=0:raise ValueError('Nonzero final exploration floor')
    for p,h in done['files'].items():
        if sha(out/p)!=h:raise ValueError('Changed result '+p)
    for d in ('dclm','fwe'):check_arrays(dict(np.load(out/d/'summary.npz')),c['train']['model']['length'])
    return done

def report(base,backup):
    base=Path(base);backup=Path(backup);arrays={};done={};files=[]
    controls=json.loads((base/'inputs/complete.json').read_text())['controls']
    for name in CONTROL_NAMES:
        old=Path(controls[name]);replay_run(old)
        target=base/'runs'/name
        if target.exists():raise ValueError('Control copy already exists')
        for p in old.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.jsonl','.npz'):
                q=target/p.relative_to(old);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
                if sha(p)!=sha(q):raise ValueError('Control copy changed')
    for name in NAMES:
        run=base/'runs'/name;done[name]=replay_run(run) if name in NEW_NAMES else json.loads((run/'complete.json').read_text())
        for d in ('dclm','fwe'):arrays[name,d]=dict(np.load(run/d/'summary.npz'))
        files += [p for p in run.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.npz')]
    result=calculate(arrays);result.update(runs=done,node_replay='PASS');write_json(base/'REPORT.json',result)
    with (base/'RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(result['rows'][0]));w.writeheader();w.writerows(result['rows'])
    lines=['# Scheduled mixture exposure: matched controls and hard projection','','| Run | Domain | Mode | NLL |','|---|---|---|---:|']
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
    if len(copies)!=9:raise ValueError('Backup inventory mismatch')
    write_json(base/'backup.json',dict(archive_sha256=sha(archive),bytes=archive.stat().st_size,files=members,node_copies=copies,durability='Same-node copies; weights and full state node-only'))
    return result
