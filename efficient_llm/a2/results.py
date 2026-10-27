"""Resolve retry provenance and report exactly the registered contrasts."""
import math
import statistics
from .protocol import DATASETS, CAPS, Run, matrix, choose_lr, length_decision, lr_state_decision


def selected_lrs(rows):
    result={}
    for ds in DATASETS:
        for arm in ('P300','P1T','S72','X72','S188'):
            grid=[r for r in rows if r['dataset']==ds and r['arm']==arm
                  and r['budget']=='0.03B' and not r.get('rerun_of') and not r.get('suffix')]
            if len(grid)==3: result[ds,arm]=choose_lr(grid)
    return result


def report(rows,parents,*,include_secondary=True):
    """parents keyed 'dataset/parent', with final loss; failed required rows -> None."""
    ids=[r['run_id'] for r in rows]
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate run results')
    chosen=selected_lrs(rows)
    by_id={r['run_id']:r for r in rows}
    def effective(ds,arm,s,budget,suffix=''):
        lr=chosen.get((ds,arm))
        if lr is None:return None
        requested=Run(ds,arm,s,budget,lr,suffix)
        r=by_id.get(requested.run_id)
        if not r:return None
        if r.get('diverged'):
            retries=[v for v in rows if v.get('rerun_of')==r['run_id']]
            if len(retries)>1: raise ValueError('More than one retry')
            if not retries:return None
            retry=retries[0]
            if retry['lr']!=r['lr']/2 or any(retry[k]!=r[k] for k in ('dataset','arm','set','budget','parent','teacher')):
                raise ValueError('Invalid retry provenance')
            r=retry
        if r['status']!='complete' or r.get('diverged'):return None
        return r
    def val(ds,arm,s,budget,key='val_post'):
        r=effective(ds,arm,s,budget)
        return r[key] if r and r.get(key) is not None and math.isfinite(r[key]) else None
    def diff(a,b):return a-b if a is not None and b is not None else None
    def descriptive(values):
        defined=[v for v in values if v is not None and math.isfinite(v)]
        return dict(mean=statistics.mean(defined) if defined else None,
                    defined=len(defined),total=len(values),
                    negative=sum(v<0 for v in defined),positive=sum(v>0 for v in defined))
    def length(early,late):
        contrasts=lambda sets,budget,key: {d:[diff(val(d,late,s,budget,key),val(d,early,s,budget,key)) for s in sets] for d in DATASETS}
        main=contrasts(('S1','S2'),'0.2B','delta_post')
        pre=contrasts(('S1','S2'),'0.2B','delta_pre')
        short=contrasts(('S1','S2','S3','S4'),'0.05B','delta_post')
        return dict(**length_decision(main,pre,sum(short.values(),[])),main=main,pre=pre,short=short)
    a=length('P300','P1T'); b=length('S72','S188')
    contrasts=[]
    for d in DATASETS:
        for s in ('S1','S2'):
            x=val(d,'X72',s,'0.2B'); p=val(d,'P300',s,'0.2B'); stable=val(d,'S72',s,'0.2B')
            contrasts.append(dict(dataset=d,set=s,A_init=diff(x,p),A_teach=diff(stable,x),A=diff(stable,p)))
    lr=lr_state_decision([r['A_init'] for r in contrasts],[chosen.get((d,'X72')) for d in DATASETS])
    for item in contrasts:
        d=item['dataset']
        gain=diff(parents.get(f'{d}/S72',{}).get('val'),parents.get(f'{d}/P300',{}).get('val'))
        item.update(A_init_teacher_normalized=item['A_init'],
                    A_teach_teacher_normalized=diff(item['A_teach'],gain),
                    A_teacher_normalized=diff(item['A'],gain))
    short_a=[diff(val(d,'S72',s,'0.05B'),val(d,'P300',s,'0.05B')) for d in DATASETS for s in ('S1','S2','S3','S4')]
    curves={p:{h:val('nem',p,'S1',h,'delta_post') for h in ('0.05B','0.2B','0.8B')} for p in ('P300','P1T')}
    tables={d:{s:{p:{k:val(d,p,s,'0.2B',k) for k in ('val_post','delta_pre','delta_post')}
                   for p in ('S72','P300','S188','P1T')} for s in ('S1','S2')} for d in DATASETS}
    noise={suffix:diff((effective('nem','P300','S1','0.05B',suffix) or {}).get('val_post'),
                      val('nem','P300','S1','0.05B')) for suffix in ('-t14a','-t14b')}
    grid_gap={d:diff(val(d,'X72','S1','0.03B'),val(d,'P300','S1','0.03B')) for d in DATASETS}
    t15={}
    for d in DATASETS:
        loss=lambda p:parents.get(f'{d}/{p}',{}).get('val')
        gain=diff(loss('S72'),loss('P300')); stable_gain=diff(loss('S72'),loss('S188'))
        t15[d]=dict(annealing_gain=gain,stable_gain=stable_gain,
            annealing_gain_below_002=gain is not None and gain<.02,
            nonpositive_stable_gain=stable_gain is not None and stable_gain<=0,
            lr_history='documented by WSD schedule/iteration; consult separate logged-LR receipt if available')
    result=dict(T15=t15,phase0A=a,phase0B_length=b,phase0B_lr=dict(lr,contrasts=contrasts,short_A=short_a,
            grid_A_init=grid_gap,
            descriptive={key:descriptive([item[key] for item in contrasts]) for key in ('A_teach','A')},
            short_A_descriptive=descriptive(short_a),
            X72_P300_same_lr={d:(chosen[d,'X72']==chosen[d,'P300'] if chosen.get((d,'X72')) is not None
                and chosen.get((d,'P300')) is not None else None) for d in DATASETS}),
        nem_healing_curves=curves,comparison_tables=tables,T14_differences=noise,
        chosen_lrs={f'{d}/{arm}':v for (d,arm),v in chosen.items()},
        annealing_gain={d:diff(parents.get(f'{d}/S72',{}).get('val'),parents.get(f'{d}/P300',{}).get('val')) for d in DATASETS},
        descriptive_slope_difference=(a['E']/math.log(1000/300)-b['E']/math.log(776.2/297.3))
            if a['E'] is not None and b['E'] is not None else None,
        scope='Screening only; dependent contrasts, public checkpoint/teacher/recipe confounds. GO is not a novelty claim.',
        run_count=len(rows),failed=[r['run_id'] for r in rows if r.get('diverged') or r['status']!='complete'])
    result['parent_length_gap']={d:diff(parents.get(f'{d}/P1T',{}).get('val'),
                                      parents.get(f'{d}/P300',{}).get('val')) for d in DATASETS}
    result['log_length_effects']=dict(annealed=a['E']/math.log(1000/300) if a['E'] is not None else None,
                                    stable=b['E']/math.log(776.2/297.3) if b['E'] is not None else None)
    if include_secondary:
        # Keep tuning losses, selected LRs and original divergence decisions.
        # Secondary results only score the SAME effective runs; never reselect.
        secondary_rows=[]
        for row in rows:
            projected=dict(row)
            for name in ('val_post','delta_pre','delta_post'):
                projected[name]=row.get(name.replace('_','_secondary_',1))
            secondary_rows.append(projected)
        secondary_parents={k:dict(v,val=v.get('secondary',{}).get('loss')) for k,v in parents.items()}
        secondary=report(secondary_rows,secondary_parents,include_secondary=False)
        def quantities(item):
            return {k:v for k,v in item.items() if k not in ('label','clauses','stop_reason')}
        result['secondary']=dict(note='Descriptive only; no decision label, LR selection or divergence test.',
            phase0A=quantities(secondary['phase0A']),phase0B_length=quantities(secondary['phase0B_length']),
            phase0B_lr=quantities(secondary['phase0B_lr']),
            parent_length_gap=secondary['parent_length_gap'],comparison_tables=secondary['comparison_tables'],
            nem_healing_curves=secondary['nem_healing_curves'],T14_differences=secondary['T14_differences'])
    return result


def estimate_workload(costs):
    """Measured whole-workload estimate; no hidden old PFLOP assumption."""
    required=('step_seconds','evaluation_seconds','measurement_seconds','load_save_seconds','retry_allowance_seconds')
    out={}
    for phase,(trigger,cap) in CAPS.items():
        c=costs[phase]
        if set(c)!=set(required) or any(not math.isfinite(v) or v<0 for v in c.values()) or c['step_seconds']==0:
            raise ValueError('Specify all measured workload categories')
        runs=[r for r in matrix() if r.part==phase]
        seconds=sum(r.steps for r in runs)*c['step_seconds']+sum(c[k] for k in required[1:])
        out[phase]=dict(node_hours=seconds/3600,report_required=seconds/3600>trigger,
                        exceeds_cap=seconds/3600>cap,cap_hours=cap,runs=len(runs),costs=c)
    return out


def write_report(directory,rows,parents):
    """Small durable tables and required figures; never choose further experiments."""
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .data import write_json
    if isinstance(parents,list):parents={f'{p["dataset"]}/{p["parent"]}':p for p in parents}
    parents={key:dict(value,val=value.get('val',value.get('final',{}).get('loss'))) for key,value in parents.items()}
    result=report(rows,parents)
    root=Path(directory);root.mkdir(parents=True,exist_ok=False)
    write_json(root/'decision_phase0A.json',result['phase0A'])
    write_json(root/'decision_phase0B.json',dict(length=result['phase0B_length'],lr_state=result['phase0B_lr']))
    write_json(root/'summary.json',result)
    lines=['# A2 pilot results','',result['scope'],'',
           '| Contrast | Label | Mean (nats) |','|---|---|---|']
    for label,item in (('Annealed length',result['phase0A']),('Stable length',result['phase0B_length'])):
        lines.append(f'| {label} | {item["label"]} | {item["E"]} |')
    lines.append(f'| Initialization at fixed teacher | {result["phase0B_lr"]["label"]} | {result["phase0B_lr"]["mean"]} |')
    lines.extend(['','| Dataset | Annealed E_d | Stable E_d |','|---|---|---|'])
    for ds in DATASETS:lines.append(f'| {ds} | {result["phase0A"]["E_d"][ds]} | {result["phase0B_length"]["E_d"][ds]} |')
    for ds,tables in result['comparison_tables'].items():
        for layer_set,arms in tables.items():
            lines.extend(['',f'### {ds}, {layer_set}', '', '| State | Short parent: val / Δpre / Δpost | Long parent: val / Δpre / Δpost |','|---|---|---|'])
            for state,short,long in (('Stable','S72','S188'),('Annealed','P300','P1T')):
                cell=lambda arm: ' / '.join('NA' if arms[arm][key] is None else f'{arms[arm][key]:.6f}' for key in ('val_post','delta_pre','delta_post'))
                lines.append(f'| {state} | {short}: {cell(short)} | {long}: {cell(long)} |')
    lines.extend(['','See summary.json for every clause, missing input, chosen LR, T14 difference and normalized contrast.',
        'Common Crawl overlap with parent pretraining is possible. Secondary sets may overlap more for longer-trained parents.',
        'Timing, test receipts and deviations must accompany this report before scientific interpretation.'])
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    def save(fig,name):fig.tight_layout();fig.savefig(root/name,dpi=150);plt.close(fig)
    for ds in DATASETS:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for arm in ('P300','P1T','S72','S188','X72'):
            for ax,s in zip(axes,('S1','S2')):
                # Use only the selected, successful effective runs; do not plot
                # original divergences or LR-grid fits as main learning curves.
                chosen=result['chosen_lrs'].get(f'{ds}/{arm}')
                good=[]
                for r in rows:
                    if (r['dataset'],r['arm'],r['set'])!=(ds,arm,s) or r['budget']=='0.03B' or 't14' in r.get('suffix',''):continue
                    if r.get('diverged') or r['status']!='complete':continue
                    if not r.get('rerun_of') and r['lr']!=chosen:continue
                    good.append(r)
                good.sort(key=lambda r:r['heal_tokens'])
                if good:ax.plot([r['heal_tokens']/1e9 for r in good],[r['delta_post'] for r in good],marker='o',label=arm)
        for ax,s in zip(axes,('S1','S2')):ax.set(title=f'{ds}: {s}',xlabel='Healing tokens (B)',ylabel='Post-healing damage (nats)')
        if axes[0].lines:axes[0].legend()
        save(fig,f'{ds}_healing.png')
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        fields=('bypass_delta','layer_bi','relative_update')
        plotted=False
        for p in ('P300','P1T','S72','S188'):
            item=parents.get(f'{ds}/{p}',{})
            for ax,field in zip(axes,fields):
                if field in item:ax.plot(range(len(item[field])),item[field],label=p);plotted=True
        for ax,field in zip(axes,fields):ax.set(title=f'{ds}: {field}',xlabel='Original layer index')
        if plotted:
            axes[0].legend();save(fig,f'{ds}_profiles.png')
        else:plt.close(fig)
    return result
