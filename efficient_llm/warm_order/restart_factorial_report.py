"""Factorial contrasts for optimizer-state and LR-policy effects on dense retention."""
import csv,json,shutil,tarfile
from pathlib import Path
import numpy as np
from .data import sha
from .evaluate import paired_ci
from .pair_order_report import check_arrays
from .strength_refinement import check
from .recursion_eval import check_train
from .train import write_json

NAMES=[f'{a}-s{s}' for a in ('FL','IH','IL') for s in (0,1)]
ALL=NAMES+[f'{a}-s{s}' for a in ('FH','P') for s in (0,1)]
FLAGS=['reused_final_exploratory','multiple_comparisons','two_seeds','repeated_body_rows',
       'conditional_suffix480','LR_policy_not_peak_only','same_candidate10_not_exact_compute',
       'historical_FH_runtime_sensitive','diagnostic_not_order_win','weights_node_only']


def calculate(arrays):
    rows=[];comparisons={};signals={a:[] for a in ('FL','IH','IL')}
    for name in ALL:
        for domain in ('dclm','fwe'):
            a=arrays[name,domain];anchor=arrays[f'FH-s{name[-1]}',domain]
            np.testing.assert_array_equal(a['ids'],anchor['ids']);np.testing.assert_array_equal(a['clusters'],anchor['clusters'])
            for mode in ('soft','hard'):
                rows.append(dict(name=name,domain=domain,metric=mode,nll=float(a[mode].mean(dtype=np.float64)),sequences=len(a[mode])))
                comparisons[f'{name}/{domain}/{mode}-minus-parent']=paired_ci(a[mode]-arrays[f'P-s{name[-1]}',domain][mode],a['clusters'])
    for seed in (0,1):
        for domain in ('dclm','fwe'):
            c={arm:arrays[f'{arm}-s{seed}',domain] for arm in ('FH','FL','IH','IL')}
            for mode in ('soft','hard'):
                for left,right in [('FH','FL'),('IH','IL'),('FH','IH'),('FL','IL'),('FH','IL')]:
                    comparisons[f's{seed}/{domain}/{mode}/{left}-minus-{right}']=paired_ci(c[left][mode]-c[right][mode],c[left]['clusters'])
                d=(c['FH'][mode]-c['FL'][mode])-(c['IH'][mode]-c['IL'][mode])
                comparisons[f's{seed}/{domain}/{mode}/interaction']=paired_ci(d,c['FH']['clusters'])
            for arm in signals:
                signals[arm].append(float((c['FH']['soft']-c[arm]['soft']).mean(dtype=np.float64))>=.005)
    return dict(rows=rows,comparisons=comparisons,retention_signal={a:all(v) for a,v in signals.items()},flags=FLAGS)


def replay_run(run):
    run=Path(run);c=json.loads((run/'config.json').read_text());done=json.loads((run/'complete.json').read_text())
    check(c['hashes']);td=check_train(run/'train');p=done['proof']
    recipe=json.loads((run/'train/recipe.json').read_text());tc=c['train']
    expected=tc['steps']+(recipe['parent_optimizer']['cumulative_adam_steps'] if tc['factor'].startswith('I') else 0)
    if td['factor']!=tc['factor'] or td['cumulative_adam_steps']!=expected:raise ValueError('Terminal factorial state mismatch')
    if (done['status']!='complete' or p['causal_max']!=0 or not p['gate_causality'] or
            not p['initial']['hard_parent_exact'] or
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
            arrays[name,d]=dict(np.load(dest));check_arrays(arrays[name,d],prepared['length'],parent=name.startswith('P-'));files.append(dest)
    result=calculate(arrays);result.update(runs=runs,node_replay='PASS');write_json(base/'REPORT.json',result)
    with (base/'RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(result['rows'][0]));w.writeheader();w.writerows(result['rows'])
    lines=['# Dense restart factorial','','| Run | Domain | Mode | NLL |','|---|---|---|---:|']
    lines += [f"|{r['name']}|{r['domain']}|{r['metric']}|{r['nll']:.6f}|" for r in result['rows']]
    lines += ['',f"Retention signal versus FH: {result['retention_signal']}",'',', '.join(FLAGS)]
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
