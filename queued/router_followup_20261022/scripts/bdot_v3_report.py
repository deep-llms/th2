"""Summarize the fixed v3 batch without changing its candidate pool or training."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.make_bdot_pilot import V3_RUNS


def select_candidate(rows):
    """Apply the predeclared contingency and 0.01-nat simplicity tie rule."""
    dropped_pf=rows['V2']['loss']-rows['V1']['loss']>=.01
    order=['V1' if dropped_pf else 'V2','V3','V4']
    eligible=[a for a in order if not rows[a]['spike'] and rows[a]['off_default']>=.05]
    if not eligible:return None,dropped_pf
    best=min(rows[a]['loss'] for a in eligible)
    return next(a for a in eligible if rows[a]['loss']-best<.01),dropped_pf


def gate(candidate, control, dense):
    if len(candidate)!=2 or len(control)!=2 or len(dense)!=2:
        return dict(status='pending_two_seed_confirmation')
    mean=lambda rows,key:sum(r[key] for r in rows)/len(rows)
    margins=dict(no_spike=not any(r['spike'] for r in candidate),
        off_default=mean(candidate,'off_default')>=.05,
        default_gain=mean(candidate,'default_gain')>=.01,
        chance_gain=mean(control,'loss')-mean(candidate,'loss')>=.01,
        dense_gap=mean(candidate,'loss')-mean(dense,'loss')<=.01,
        throughput=mean(candidate,'throughput_ratio')>=.6)
    return dict(status='pass' if all(margins.values()) else 'fail',criteria=margins,
        mean_loss=mean(candidate,'loss'),gap_to_dense=mean(candidate,'loss')-mean(dense,'loss'),
        gain_over_chance=mean(control,'loss')-mean(candidate,'loss'),
        default_gain=mean(candidate,'default_gain'))


def report(prepared,run_dir,verification,output,pilot,random_run):
    read=lambda p:json.loads(Path(p).read_text())
    checks=read(verification);speeds=checks['throughput']
    old={arm:read(root/'prepared/runs'/arm/'final_eval.json') for arm,root in [('D',pilot),('R',random_run)]}
    rows=[]
    for arm,seed in V3_RUNS:
        name=f'{arm}-s{seed}';train=prepared/'runs'/name
        result=read(run_dir/name/'result.json');study=result['route_study']
        original=read(train/'final_eval.json');verified=read(run_dir/(name+'.verified.json'))
        if (result['global_step']!=4768 or verified['global_step']!=4768
                or abs(result['loss']-original['eval_loss'])>1e-5
                or result['num_tokens']!=old['D']['eval_num_tokens']):
            raise ValueError('Final evaluation/step mismatch: '+name)
        usage=study['route_diagnostics']; health=study['router_health']
        logs=[json.loads(line) for line in (train/'metrics.jsonl').read_text().splitlines() if line]
        spike=any(r.get('eval_spike',False) for r in logs)
        losses={k:v['loss'] for k,v in study['interventions'].items()}
        rows.append(dict(arm=arm,seed=seed,name=name,status='complete',loss=result['loss'],
            ppl=result['perplexity'],gap_to_D=result['loss']-(old['D']['eval_loss'] if seed==0 else
                read(prepared/'runs/D-s1/final_eval.json')['eval_loss']),
            off_default=result['off_default_share'],distinct=usage['mean_distinct_modules'],
            minimum_load=min(usage['module_load_share']),maximum_load=max(usage['module_load_share']),
            saturated_share=health['saturated_share'],saturated_eligible_share=health['saturated_eligible_share'],
            spike=spike,default_gain=study['default_path_loss_increase'],interventions=losses,
            throughput_ratio=speeds[arm]['native_ratio'],attempts=verified.get('attempts',[])))
    zero={r['arm']:r for r in rows if r['seed']==0}
    selected,drop_pf=select_candidate(zero)
    controls={'V1':'C1','V2':'C2','V3':'C3','V4':'R'}
    dense0=dict(loss=old['D']['eval_loss'])
    controls_rows=[r for r in rows if selected and r['arm']==controls[selected]]
    if selected=='V4':controls_rows.append(dict(loss=old['R']['eval_loss']))
    decision=(gate([r for r in rows if r['arm']==selected],controls_rows,
                   [dense0,*[r for r in rows if r['arm']=='D']]) if selected else dict(status='fail_no_eligible_candidate'))
    needed=[]
    if drop_pf:needed.append('C1 seed 0 (random/no-reuse/r1/no PF) under the predefined PF contingency')
    if selected in ('V1','V3','V4'):
        needed.append(f'Check section 6.2 seed-0 margins for {selected}; candidate/control seed 1 required before judgment')
    if decision['status']=='pending_two_seed_confirmation':
        decision['warning']='No candidate may pass without the specified two-seed comparison.'
    old_loss={name:r['original_loss'] for name,r in checks['carryover'].items()}
    comparisons={f'{a} minus {b}':zero[a]['loss']-(zero[b]['loss'] if b in zero else old_loss[b])
        for a,b in [('V1','N-A'),('V2','V1'),('V3','V2'),('V4','V1'),
                    ('V4','P1-A'),('V4','P1-A-prime'),('V2','C2'),('V3','C3'),('V4','R'),
                    ('V5','V4'),('V6','V2'),('V7','V2')]}
    closure=zero['P1-A-prime']
    closure_criteria=dict(no_spike=not closure['spike'],off_default=closure['off_default']>=.05,
        default_gain=closure['default_gain']>=.01,chance_gain=old_loss['R']-closure['loss']>=.01,
        dense_gap=closure['loss']-old_loss['D']<=.01,throughput=closure['throughput_ratio']>=.6)
    result=dict(status='ok',rows=rows,selected=selected,placement_film_contingency=drop_pf,
        comparisons=comparisons,closure_seed0_criteria=closure_criteria,
        gate=decision,followup_decisions=needed,
        scope='V5/V6/V7 are exploratory and excluded from the predefined gate candidate pool.')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as handle:json.dump(result,handle,indent=2)
    lines=['# B-DOT v3 batch results','',
        'Loss/off-default: full frozen validation. Usage/saturation: first 4,096 sequences. '
        'Saturation includes sole-candidate choices; eligible-only values are in the JSON.', '',
        '| Run | Loss | Gap to same-seed D | Off-default | Distinct | Min/max load | Saturated |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["name"]} | {r["loss"]:.6f} | {r["gap_to_D"]:+.6f} | '
            f'{100*r["off_default"]:.2f}% | {r["distinct"]:.4f} | '
            f'{100*r["minimum_load"]:.2f}% / {100*r["maximum_load"]:.2f}% | {100*r["saturated_share"]:.2f}% |')
    lines+=['','| Run | Own loss | Default loss | Random loss |','|---|---:|---:|---:|']
    for r in rows:
        i=r['interventions'];lines.append(f'| {r["name"]} | {i["as_trained"]:.6f} | {i["fixed"]:.6f} | {i["random"]:.6f} |')
    lines+=['','| Comparison | Loss difference |','|---|---:|',
            *[f'| {k} | {v:+.6f} |' for k,v in comparisons.items()]]
    lines+=['',f'Selected: {selected}. Gate: {decision["status"]}.',*needed,'',result['scope']]
    output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print('V3_BATCH_REPORT '+json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','run-dir','verification','output','pilot','random-run'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();report(a.prepared,a.run_dir,a.verification,a.output,a.pilot,a.random_run)


if __name__=='__main__':main()
