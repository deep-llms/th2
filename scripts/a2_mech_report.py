"""CPU-only readout of registered mechanism probes, with explicit missing clauses."""
import json
from pathlib import Path
import numpy as np
from scripts.a2_quant_io import write,lock
from efficient_llm.a2.quant_stats import PARENTS,CONTRASTS,REFERENCES,nll_report
from efficient_llm.a2.mech import token_buckets

DATASETS=('dclm','fwe','nem')


def relative(values):
    result={}
    for ds in DATASETS:
        row=values.get(ds,{})
        if any(row.get(p) is None for p in PARENTS):continue
        arr=np.array([row[p] for p in PARENTS]);table={}
        for c in ('L_ann','L_stab','R_short'):
            v=float(arr@CONTRASTS[c]);ref=row[REFERENCES[c]]
            table[c]=dict(contrast=v,reference=REFERENCES[c],reference_value=ref,relative=v/ref if ref else None)
        result[ds]=table
    return result


def patterns(table):
    clauses={}
    for ds in DATASETS:
        r=table.get(ds,{})
        a={k:r.get(k,{}).get('relative') for k in ('L_ann','L_stab','R_short')}
        valid=all(v is not None and np.isfinite(v) for v in a.values())
        clauses[ds]=dict(L_ann_positive=a['L_ann']>0 if valid else None,
            L_stab_positive=a['L_stab']>0 if valid else None,
            L_ann_above_R_short=a['L_ann']>a['R_short'] if valid else None,
            L_stab_above_R_short=a['L_stab']>a['R_short'] if valid else None,
            R_short_positive=a['R_short']>0 if valid else None,
            R_short_above_L_stab=a['R_short']>a['L_stab'] if valid else None)
    def decide(names):
        vals=[clauses[ds][k] for ds in DATASETS for k in names]
        return None if any(v is None for v in vals) else all(vals)
    return dict(clauses=clauses,pruning=decide(('L_ann_positive','L_stab_positive','L_ann_above_R_short','L_stab_above_R_short')),
                ptq=decide(('R_short_positive','R_short_above_L_stab')))


def summarize(cfg,root):
    root=Path(root);sha=cfg['spec_sha256'];metrics={};weights={};layers={};tests=[];coverage={};noise={};seed_values={};arrays={}
    if (root/'preparation_checks.json').exists():tests.extend(json.loads((root/'preparation_checks.json').read_text()))
    def put(metric,ds,p,value):metrics.setdefault(metric,{}).setdefault(ds,{})[p]=value
    for ds in DATASETS:
        for p in PARENTS:
            key=ds+'/'+p
            for phase in ('weights','layers','noise','tokens'):
                folder=root/'workers'/phase/ds/p
                path=folder/'result.json'
                if not path.exists():continue
                row=json.loads(path.read_text());tests.extend(row['checks']);r=row['results']
                coverage[phase+'/'+key]=True
                if phase=='weights':
                    weights[key]=r
                    for name,v in r['means'].items():put(name,ds,p,v)
                elif phase=='layers':
                    layers[key]=r
                    for name in ('BI_S1','S1_sum','rho'):put(name,ds,p,r.get(name))
                    for field in ('BI','update','single_damage'):
                        for index,v in enumerate(r.get(field,[])):put(field+'_'+str(index),ds,p,v)
                elif phase=='noise':
                    baseline=folder/'parent-final.json'
                    if not baseline.exists():continue
                    ref=json.loads(baseline.read_text())['per_sequence_nll'];noise.setdefault(ds,{}).setdefault('parent',{})[p]=ref
                    arrays.setdefault(ds,{})[p]=np.array(ref)
                    for name in ('N-iso-0.1','N-iso-0.2','N-iso-0.3','N-rtn3'):
                        vals=[]
                        for seed in (0,1):
                            f=folder/(name+'-seed'+str(seed)+'.json')
                            if not f.exists():continue
                            ar=np.array(json.loads(f.read_text())['per_sequence_nll']);vals.append(ar)
                            damage=float((ar-ref).mean());seed_values.setdefault(name,{}).setdefault(str(seed),{}).setdefault(ds,{})[p]=damage
                        if len(vals)==2:
                            mean=np.mean(vals,axis=0);noise[ds].setdefault(name,{})[p]=mean.tolist();put(name,ds,p,float((mean-ref).mean()))
    write(root/'weights.json',dict(spec_sha256=sha,parents=weights))
    write(root/'layers.json',dict(spec_sha256=sha,parents=layers))
    paired=nll_report(noise);write(root/'noise_paired.json',dict(spec_sha256=sha,**paired))
    readout={m:dict(values=v,relative_contrasts=relative(v),patterns=patterns(relative(v))) for m,v in metrics.items()}
    seed_readout={m:{s:patterns(relative(v)) for s,v in seeds.items()} for m,seeds in seed_values.items()}
    for m,seeds in seed_readout.items():
        seed_readout[m]['ordering_flips']={ds:{k:seeds['0']['clauses'][ds][k]!=seeds['1']['clauses'][ds][k]
            if seeds['0']['clauses'][ds][k] is not None and seeds['1']['clauses'][ds][k] is not None else None
            for k in seeds['0']['clauses'][ds]} for ds in DATASETS}
    from scripts.a2_remaining import read_locked
    quant=read_locked(cfg['quant_decision']);correlations={};ratios={}
    for m,v in metrics.items():
        correlations[m]={}
        for c in ('pre-S1','rtn3','gptq3'):
            try:
                x=np.array([[v[d][p] for p in PARENTS] for d in DATASETS],dtype=float)
                y=np.array([[quant['datasets'][d][c]['damage'][p]['damage'] for p in PARENTS] for d in DATASETS])
                x-=x.mean(1,keepdims=True);y-=y.mean(1,keepdims=True)
                corr=float(np.corrcoef(x.ravel(),y.ravel())[0,1])
                correlations[m][c]=corr if np.isfinite(corr) else None
            except (KeyError,TypeError):correlations[m][c]=None
    for ds in DATASETS:
        for p in PARENTS:
            v=metrics.get('N-rtn3',{}).get(ds,{}).get(p);denom=quant['datasets'][ds]['rtn3']['damage'][p]['damage']
            ratios[ds+'/'+p]=v/denom if v is not None and denom else None
    buckets={};he={};token_inventory=[]
    for ds in ('dclm','fwe'):
        values={}
        for path in sorted((root/'workers/tokens'/ds).glob('*/*.npy')):
            values[path.stem]=np.load(path,mmap_mode='r');token_inventory.append(lock(path))
        try:
            from efficient_llm.a2.data import PackedRows
            rows=PackedRows(cfg['data_root'],'final',verify=False);target=np.asarray(rows.ids)[:,1:]
            full_frequency=np.bincount(np.asarray(rows.ids).ravel())
            table=token_buckets(values,target,frequency_counts=full_frequency)
            # Paired sequence bootstrap of bucket gaps and retention; bucket membership fixed.
            n=len(target);draws=np.random.default_rng(0).multinomial(n,np.full(n,1/n),size=10000)
            flat=target.ravel();freq=full_frequency[flat]
            for kind,key in [('frequency',freq),('difficulty',values['S188'].ravel())]:
                retention_draws=[]
                for b,ix in zip(table['buckets'][kind],np.array_split(np.argsort(key,kind='stable'),10),strict=True):
                    seq=ix//target.shape[1];counts=np.bincount(seq,minlength=n);den=draws@counts
                    deltas={
                        'parent_gap':values['P300'].ravel()[ix].astype(float)-values['P1T'].ravel()[ix],
                        'student_gap':values['student-P300'].ravel()[ix].astype(float)-values['student-P1T'].ravel()[ix],
                        'pre_extra':(values['pre-P1T'].ravel()[ix].astype(float)-values['P1T'].ravel()[ix])-(values['pre-P300'].ravel()[ix].astype(float)-values['P300'].ravel()[ix]),
                        'A_init':values['student-X72'].ravel()[ix].astype(float)-values['student-P300'].ravel()[ix]}
                    with np.errstate(divide='ignore',invalid='ignore'):
                        samples={k:(draws@np.bincount(seq,weights=v,minlength=n))/den for k,v in deltas.items()}
                    b['empty_bucket_resamples']=int((den==0).sum())
                    b['ci95']={k:np.quantile(v,[.025,.975]).tolist() if np.isfinite(v).all() else None for k,v in samples.items()}
                    with np.errstate(divide='ignore',invalid='ignore'):r=samples['student_gap']/samples['parent_gap']
                    b['ci95']['retention']=np.quantile(r,[.025,.975]).tolist() if np.isfinite(r).all() else None
                    b['bootstrap_denominator_crosses_zero']=bool(samples['parent_gap'].min()<=0<=samples['parent_gap'].max())
                    retention_draws.append(r)
                r=[b['retention'] for b in table['buckets'][kind]]
                valid=all(v is not None for v in r);low=float(np.mean(r[:3])) if valid else None;high=float(np.mean(r[-3:])) if valid else None
                effect=np.mean(retention_draws[:3],axis=0)-np.mean(retention_draws[-3:],axis=0)
                he.setdefault(ds,{})[kind]=dict(bottom3=low,top3=high,
                    clause=(low<high if kind=='frequency' else high<low) if valid else None,
                    low_minus_high_ci95=np.quantile(effect,[.025,.975]).tolist() if np.isfinite(effect).all() else None,
                    flagged_buckets=[b['bucket'] for b in table['buckets'][kind] if b['low_denominator']])
            buckets[ds]=table
        except Exception as e:
            tests.append(dict(test='token_buckets:'+ds,passed=False,detail=repr(e),spec_sha256=sha))
    write(root/'buckets.json',dict(spec_sha256=sha,datasets=buckets,readout=he,resamples=10000,seed=0))
    write(root/'token_inventory.json',dict(spec_sha256=sha,files=token_inventory))
    def pattern(m,k):return readout.get(m,{}).get('patterns',{}).get(k)
    p1=[pattern(m,'pruning') for m in ('S1_sum','BI_S1')]
    h=dict(H_P1=True if True in p1 else False if all(v is False for v in p1) else None,
        H_P2=pattern('rho','pruning'),H_Q1=pattern('rtn3_weight_error','ptq'),H_Q2=pattern('N-iso-0.2','ptq'))
    he_clauses=[he.get(d,{}).get(k,{}).get('clause') for d in ('dclm','fwe') for k in ('frequency','difficulty')]
    h['H_E']=all(he_clauses) if all(v is not None for v in he_clauses) else None
    expected={f'{phase}/{d}/{p}' for phase in ('weights','layers','noise') for d in DATASETS for p in PARENTS}|{f'tokens/{d}/{p}' for d in ('dclm','fwe') for p in PARENTS}
    missing=sorted(expected-set(coverage));nnoise=sum(len(v) for ds in noise.values() for k,v in ds.items() if k!='parent')
    complete=not missing and nnoise==48 and len(token_inventory)==18 and len(weights)==12 and all(len(v['matrices'])==168 for v in weights.values()) and all(len(v.get('single_damage',[]))==24 for v in layers.values())
    tests.append(dict(test='registered_coverage',passed=complete,detail=dict(missing_workers=missing,noise_seed_means=nnoise,token_arrays=len(token_inventory)),spec_sha256=sha))
    failed=[r for r in tests if not r['passed']]
    result=dict(spec_sha256=sha,hypotheses=h,metrics=readout,noise_seed_readout=seed_readout,N_rtn3_over_rtn3=ratios,
        correlations_dataset_centered=correlations,H_E_readout=he,coverage_complete=complete,flags=failed,
        interpretation='Associations across public length x LR-state checkpoints, not causal effects; flagged tests qualify affected results.')
    write(root/'tests.json',dict(spec_sha256=sha,tests=tests));write(root/'decision_mech.json',result)
    text='# A2 M — mechanism probes\n\nSpec SHA-256: `'+sha+'`.\n\n'
    text+='Execution coverage: '+str(complete)+'. Scientific failed checks: '+str(len(failed))+'.\n\n'
    text+='| Hypothesis | Preregistered pattern met |\n|---|---|\n'+''.join('| '+k+' | '+str(v)+' |\n' for k,v in h.items())
    text+='\nThese are associations, not causal claims. `decision_mech.json` contains all clauses, relative contrasts, seed-order flips and dataset-centered correlations. `noise_paired.json` and `buckets.json` use paired 10,000 sequence resamples, seed 0. Low-denominator buckets remain flagged.\n\n'
    text+='Weights use existing saved RTN outputs. No training or new compression. Source A2 Q QT3 was superseded by the saved-grid audit; original failures are retained there. QT6 is an accepted minor deviation; calibration sensitivity is not erased.\n\n'
    text+='Per-token arrays stay outside Git. MT5 uses the first full small-set sequence in FP32. MT3 measures generated noise before BF16 storage rounding. RNG seed = 202610310000 + 1000 * seed + matrix_index. Bucket ties use stable sequence/position order.\n\n'
    text+='Remaining/flags: '+json.dumps(failed,ensure_ascii=False)+'. Native terminal handoff must be verified separately after supervisor exit.\n'
    (root/'REPORT.md').write_text(text)
    return result
