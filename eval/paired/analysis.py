"""Reports for fixed paired suites. Positive differences favour P20."""
import math
import numpy as np
from . import artifacts, protocol, stats

HISTORY = ('Earlier few-shot accuracies, fine-tuning scores and PPL were known. '
           'The candidate suite was drafted with that knowledge. These are checkpoint-level '
           'comparisons; independent confirmation requires another pretraining seed.')


def summarize(a, b, B, seed):
    diff = artifacts.align(a,b)
    result = {}
    for j,(metric,d) in enumerate(diff.items()):
        result[metric] = dict(A=float(a['metrics'][metric].mean()),B=float(b['metrics'][metric].mean()),
                            **stats.paired(d,B,seed+j,a['clusters']))
    return result,diff


def aggregate(differences, names, metric, B, seed, clusters):
    names = [t for t in names if metric in differences[t]]
    if not names: return None
    return dict(tasks=names,weights='equal task weights; examples pooled within groups',
                **stats.aggregate([differences[t][metric] for t in names],B,seed,[clusters[t] for t in names]))


def suite(root, frozen, name='Z', legacy_f=None):
    """root/{A10,A20,P20}/{Z,F,probes,paloma}/TASK contains result files.

    legacy_f maps A20/P20 -> task -> historical directory, supplied explicitly.
    It is never used for confirmatory Suite Z.
    """
    from pathlib import Path
    root=Path(root); B=frozen['resamples']; seed=frozen['analysis_seed']
    report=dict(suite=name,protocol_sha256=frozen['protocol_sha256'],history=HISTORY,
                difference='P20 minus A20',tasks={},averages={},provenance={})
    differences={};references={};clusters={};symmetric=[];names=protocol.tasks(name)
    selected_path=None;identities={};plan=None
    if name=='Z':
        from eval.models import file_hash
        from . import audit
        plan=artifacts.load_json(root/'plan.json')
        if plan.get('plan_sha256')!=protocol.digest({k:v for k,v in plan.items() if k!='plan_sha256'}) or plan['protocol_sha256']!=frozen['protocol_sha256']:
            raise ValueError('Suite Z requires its frozen study plan')
        calibration=artifacts.load_json(plan['calibration'])
        if (not calibration['full_protocol'] or not calibration['passed'] or calibration['statistics_sha256']!=file_hash(stats.__file__)
            or calibration.get('calibration_code_sha256')!=file_hash(audit.__file__)
            or file_hash(plan['calibration'])!=plan['calibration_sha256']):raise ValueError('Suite Z calibration gate failed')
        precision_report=artifacts.load_json(root/'precision.json')
        if (precision_report['protocol_sha256']!=frozen['protocol_sha256'] or precision_report.get('plan_sha256')!=plan['plan_sha256']
            or set(precision_report['tasks'])!=set(protocol.PRECISION)):
            raise ValueError('Missing or mismatched precision gate')
        selected_path='production' if all(r['passed'] for r in precision_report['tasks'].values()) else 'reference'
        if precision_report['suite_Z_path']!=selected_path:raise ValueError('Invalid precision path decision')
        report['precision_gate']=precision_report
    for i,(task,shots) in enumerate(names.items()):
        def get(arm):
            directory=legacy_f[arm][task] if legacy_f else root/arm/name/task
            obj=artifacts.read(directory,task,shots,frozen,legacy=legacy_f is not None)
            identity={k:obj['metadata'].get(k) for k in ('checkpoint_sha256','tokenizer_sha256','precision_path','attention_backend')}
            if arm in identities and identity!=identities[arm]:raise ValueError('Mixed checkpoints/precision within a suite')
            identities[arm]=identity
            if plan is not None and obj['metadata']['checkpoint_sha256']!=plan['checkpoints'][arm]['sha256']:
                raise ValueError('Checkpoint differs from frozen study plan')
            if plan is not None and obj['metadata']['dataset_manifest_sha256']!=plan['manifest_sha256']:
                raise ValueError('Dataset differs from frozen study plan')
            return obj
        a,b=get('A20'),get('P20')
        if selected_path is not None and any(o['metadata']['precision_path']!=selected_path for o in (a,b)):
            raise ValueError('Suite Z precision path differs from gate decision')
        report['provenance'][task]={arm:obj['metadata'] for arm,obj in [('A20',a),('P20',b)]}
        clusters[task]=a['clusters']
        if name=='paloma':
            artifacts.align(a,b)
            delta=np.array([y['ll']-x['ll'] for x,y in zip(a['records'],b['records'])])/math.log(2)
            counts=np.array([r['bytes'] for r in a['records']])
            report['tasks'][task]=dict(BPB_reduction=stats.ratio(delta,counts,B,seed+i),
                A_bpb=-sum(r['ll'] for r in a['records'])/sum(counts)/math.log(2),
                P_bpb=-sum(r['ll'] for r in b['records'])/sum(counts)/math.log(2))
            continue
        table,differences[task]=summarize(a,b,B,seed+10*i)
        report['tasks'][task]=dict(metrics=table)
        pooled=(a['metrics']['ACC'].mean()+b['metrics']['ACC'].mean())/2
        n=len(a['ids']); se=math.sqrt(pooled*(1-pooled)/n)
        selected=pooled-a['chance']>=3*se
        if selected:symmetric.append(task)
        report['tasks'][task]['symmetric_filter']=dict(pooled_accuracy=float(pooled),chance=a['chance'],
            se=se,kept=bool(selected),status='descriptive only')
        if name in ('Z','probes'):
            c=get('A10'); artifacts.align(c,b)
            if selected_path is not None and c['metadata']['precision_path']!=selected_path:raise ValueError('A10 precision path differs')
            ref,references[task]=summarize(c,a,B,seed+10*i)
            report['provenance'][task]['A10']=c['metadata']
            report['tasks'][task]['reference_A20_minus_A10']=ref
            report['tasks'][task]['progress_ratio']={m:stats.progress_ratio(
                [(differences[task][m],references[task][m])],B,seed+10*i,[a['clusters']])
                for m in ('M1','M2','ACC') if m in differences[task]}
            acc=a['metrics']['ACC']; gain=references[task]['ACC'];se20=float(np.sqrt(acc.mean()*(1-acc.mean())/n))
            gse=float(gain.std(ddof=1)/np.sqrt(n))
            report['tasks'][task]['admission_description']=dict(
                accuracy_10k=float(c['metrics']['ACC'].mean()),accuracy_20k=float(acc.mean()),
                chance=a['chance'],accuracy_se=se20,above_chance_se=float((acc.mean()-a['chance'])/se20) if se20 else None,
                gain=float(gain.mean()),gain_se=gse,gain_over_se=float(gain.mean()/gse) if gse else None,
                used_for_selection=False)
        if name=='probes':
            _,counts=np.unique(a['clusters'],return_counts=True)
            report['tasks'][task]['story_audit']=dict(stories=len(counts),questions=n,
                cluster_sizes=counts.tolist(),cluster_ids=a['clusters'])
    if name=='paloma':
        report['averages']['BPB_reduction']=dict(estimate=float(np.mean([r['BPB_reduction']['estimate'] for r in report['tasks'].values()])),weights='equal sources; descriptive')
        return report
    for metric in ('M1','M2','M3','ACC'):
        report['averages'][metric]=aggregate(differences,names,metric,B,seed,clusters)
        selected=[t for t in names if metric in differences[t]]
        if selected:
            report['averages'][metric].update(A=float(np.mean([report['tasks'][t]['metrics'][metric]['A'] for t in selected])),
                B=float(np.mean([report['tasks'][t]['metrics'][metric]['B'] for t in selected])))
    if references:
        report['progress_ratios']={m:stats.progress_ratio(
            [(differences[t][m],references[t][m]) for t in names if m in differences[t]],B,seed,
            [clusters[t] for t in names if m in differences[t]]) for m in ('M1','M2','ACC')}
    if name=='Z':
        h1=report['averages']['M1'];h2=report['averages']['M2'];reject=h1['p_std']<.05
        report['confirmatory']=dict(H1_rejected=reject,H2_tested=reject,H2_rejected=reject and h2['p_std']<.05,
            status='fixed sequence; two-sided sign flips under joint pair-swap invariance',
            borderline={m:(x['std_ci'] is not None and (x['p_std']<.05)!=(x['std_ci'][0]>0 or x['std_ci'][1]<0))
                        for m,x in [('M1',h1),('M2',h2)]})
        report['symmetric_secondary']={m:aggregate(differences,symmetric,m,B,seed,clusters) for m in ('M1','M2')}
    if name=='probes':
        report['contrasts']={}
        first=protocol.PROBES[0]
        for t in protocol.PROBES[1:]:
            report['contrasts'][t+'-qa1']={m:dict(
                P20_minus_A20=stats.contrast([differences[x][m] for x in (first,t)],B,seed,[clusters[x] for x in (first,t)]),
                A20_minus_A10=stats.contrast([references[x][m] for x in (first,t)],B,seed,[clusters[x] for x in (first,t)]),
                difference_of_contrasts=stats.joint_contrast([(differences[x][m],references[x][m]) for x in (first,t)],B,seed,[clusters[x] for x in (first,t)])) for m in ('M1','M2')}
    return report


def precision(reference_a,reference_b,prod_a,prod_b,repeat_a,repeat_b,full_size):
    """Repeatability and paired-effect equivalence. No standard-deviation-only gate."""
    ref=artifacts.align(reference_a,reference_b); prod=artifacts.align(prod_a,prod_b)
    for obj in (prod_a,prod_b,repeat_a,repeat_b): artifacts.align(reference_a,obj)
    if 'M1' not in ref:raise ValueError('Precision gate requires M1 observations')
    for ref_obj,p,rep in ((reference_a,prod_a,repeat_a),(reference_b,prod_b,repeat_b)):
        if 'metadata' not in ref_obj:continue  # Pure arrays are useful for numerical unit tests.
        if any(o['metadata']['checkpoint_sha256']!=ref_obj['metadata']['checkpoint_sha256'] for o in (p,rep)):
            raise ValueError('Precision check requires identical weights across paths')
        if (ref_obj['metadata']['precision_path']!='reference' or ref_obj['metadata']['attention_backend']!='sdpa'
            or ref_obj['metadata']['weights_dtype']!='torch.float32' or ref_obj['metadata']['autocast_dtype']!='None'):
            raise ValueError('Precision reference must use FP32 SDPA')
        for key in ('precision_path','attention_backend','batch_size','seed','autocast_dtype'):
            if p['metadata'][key]!=rep['metadata'][key]:raise ValueError('Repeatability settings differ')
    ra=artifacts.align(prod_a,repeat_a);rb=artifacts.align(prod_b,repeat_b)
    output={}
    for metric in ('M1','M2'):
        if metric not in ref:continue
        d=ref[metric]; error=prod[metric]-d;n=len(d)
        if full_size<n or n<2: raise ValueError('Invalid precision sample size')
        sd=float(d.std(ddof=1)); noise=float(error.std(ddof=1))
        repeat={}
        for arm,values in [('A20',ra[metric]),('P20',rb[metric])]:
            maximum=float(np.abs(values).max())
            repeat[arm]=dict(mean=float(values.mean()),sd=float(values.std(ddof=1)),
                rmse=float(np.sqrt(np.square(values).mean())),max_abs=maximum,passed=maximum<=.01*sd)
        upper=float(abs(error.mean())+2*noise/math.sqrt(n));threshold=.10*sd/math.sqrt(full_size)
        output[metric]=dict(N=full_size,n=n,reference_sd=sd,error_mean=float(error.mean()),error_sd=noise,
            variance_passed=noise<=.10*sd,bias_upper=upper,bias_threshold=threshold,bias_passed=upper<=threshold,
            repeatability=repeat,passed=noise<=.10*sd and upper<=threshold and all(v['passed'] for v in repeat.values()))
    return dict(metrics=output,passed=all(v['passed'] for v in output.values()),
                scope='diagnostic for the frozen five-task sample, not all-task numerical proof')
