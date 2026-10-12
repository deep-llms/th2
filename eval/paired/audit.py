"""Real-data audits and reproducible (potentially lengthy) inference calibration."""
import math
import numpy as np
from . import artifacts, protocol, stats


def data(root,manifest,frozen):
    from lm_eval.tasks import get_task_dict
    from eval.benchmarks import local_dataset_paths
    from eval.models import file_hash
    mapping=local_dataset_paths(root,manifest);out={}
    for name in protocol.CUSTOM:
        task=get_task_dict(protocol.configs(name,mapping),task_manager=protocol.manager())[name]
        rows=list(task.eval_docs)
        if not rows:raise ValueError('Empty custom task')
        for row in rows:
            choices=task.doc_to_choice(row);gold=int(task.doc_to_target(row))
            if not 0<=gold<len(choices) or any(not isinstance(c,str) or not c for c in choices):
                raise ValueError('Invalid custom choice or target')
            if name=='commonsense_qa_cf' and len(choices)!=5:raise ValueError('CommonsenseQA needs five choices')
            if name in protocol.PROBES and choices[gold]!=row['target'].strip():raise ValueError('Unexpected bAbI location target')
        entry=dict(rows=len(rows),task_config_sha256=protocol.digest(protocol.configs(name)))
        if name in protocol.PROBES:
            groups=artifacts.story_clusters([r['input'] for r in rows]);_,sizes=np.unique(groups,return_counts=True)
            entry.update(stories=len(sizes),cluster_sizes=sizes.tolist(),cluster_ids=groups)
            task.set_fewshot_seed(42)
            context=task.fewshot_context(rows[0],num_fewshot=2)
            demonstrations=task.config.fewshot_config.samples
            if len(demonstrations)!=2 or any(task.doc_to_text(demo) not in context or demo['target'] not in context for demo in demonstrations):
                raise ValueError('Fixed bAbI demonstrations missing')
            entry['demonstrations_sha256']=protocol.digest(demonstrations)
        if name=='social_iqa_norm':
            builtin_config=protocol.configs('social_iqa')[0]
            builtin_config.update(mapping[name])
            builtin=get_task_dict([builtin_config],task_manager=protocol.manager())['social_iqa']
            for row in rows:
                if (task.doc_to_text(row)!=builtin.doc_to_text(row) or task.doc_to_choice(row)!=builtin.doc_to_choice(row)
                    or task.doc_to_target(row)!=builtin.doc_to_target(row)):
                    raise ValueError('Social IQa canonical prompt/choice/target changed')
                responses=[(-float(i+1),False) for i in range(3)]
                if task.process_results(row,responses)['acc']!=builtin.process_results(row,responses)['acc']:
                    raise ValueError('Social IQa raw scoring differs')
            entry['canonical_social_acc_equivalent']=True
        out[name]=entry
    return dict(tasks=out,status='passed',protocol_sha256=frozen['protocol_sha256'],manifest_sha256=file_hash(manifest))


def calibration_trial(args):
    case,B,seed,iteration=args
    # Independent fixed streams per case/dataset; worker count cannot change results.
    rng=np.random.default_rng(np.random.SeedSequence([seed,iteration,list(CALIBRATION_CASES).index(case)]))
    groups=None;truth=0.;truth_z=0.;task_truth=0.;task_truth_z=0.;reps=2*B if case.startswith('T4a') else B
    if case.startswith('T4a'):
        scales=[1.,.1,.02];shift=.05 if case=='T4a_effect' else 0.
        ds=[(rng.standard_t(5,n)+shift)*scale for n,scale in zip((1000,1500,2500),scales)]
        truth=shift*np.mean(scales);truth_z=shift/math.sqrt(5/3)
        task_truth=shift;task_truth_z=truth_z
    elif case=='T4b':
        ds=[(rng.lognormal(0,1,n)-math.exp(.5))*scale for n,scale in zip((500,1000,1500),(1.,.1,.02))]
    elif case=='T4c':
        ds=[rng.standard_t(5,1000)/math.sqrt(5/3)+mu for mu in (.1,-.1,0.)]
        task_truth=task_truth_z=.1
    else:
        # Same simulated data for the story-level and example-level analyses.
        rng=np.random.default_rng(np.random.SeedSequence([seed,iteration,4]))
        d=(math.sqrt(.5)*rng.normal(size=(200,1))+math.sqrt(.5)*rng.normal(size=(200,5))).ravel()
        ds=[d];groups=[np.repeat(np.arange(200),5)] if case=='T4d_story' else None
    result=stats.aggregate(ds,reps,seed+iteration,groups)
    task=stats.paired(ds[0],reps,seed+iteration,groups[0] if groups else None)
    return [task['p']<.05,result['p']<.05,result['p_std']<.05,
        result['ci'][0]<=truth<=result['ci'][1],
        result['std_ci'] is not None and result['std_ci'][0]<=truth_z<=result['std_ci'][1],
        task['ci'][0]<=task_truth<=task['ci'][1],
        task['std_ci'] is not None and task['std_ci'][0]<=task_truth_z<=task['std_ci'][1]]


CALIBRATION_CASES=('T4a','T4a_effect','T4b','T4c','T4d_story','T4d_example')


def calibration(datasets=1000,B=1000,seed=20261012,workers=1):
    """T4a-e. Default B=2000 for T4a and 1000 otherwise; CPU only.

    Smaller sizes are code smoke checks, explicitly ineligible for the study
    gate. These simulations support assumptions, not a universal FWER bound.
    """
    from concurrent.futures import ProcessPoolExecutor
    from contextlib import nullcontext
    from eval.models import file_hash
    from pathlib import Path
    if datasets<2 or B<2 or workers<1:raise ValueError('Invalid calibration sizes')
    source_hash=file_hash(Path(stats.__file__))
    rng=np.random.default_rng(seed);report={}
    with ProcessPoolExecutor(max_workers=workers) if workers>1 else nullcontext() as pool:
        for case in CALIBRATION_CASES:
            args=[(case,B,seed,i) for i in range(datasets)]
            observations=list(pool.map(calibration_trial,args,chunksize=5)) if pool else [calibration_trial(a) for a in args]
            rates=dict(zip(('task_rejection','mean_rejection','standardized_rejection','mean_coverage','standardized_coverage',
                            'task_mean_coverage','task_standardized_coverage'),np.mean(observations,axis=0).tolist()))
            expected_failure=case=='T4d_example'
            gates=[.935<=rates[k]<=.965 for k in ('mean_coverage','standardized_coverage','task_mean_coverage','task_standardized_coverage')]
            if case!='T4a_effect':gates += [.035<=rates[k]<=.065 for k in ('mean_rejection','standardized_rejection')]
            if case not in ('T4a_effect','T4c'):gates += [.035<=rates['task_rejection']<=.065]
            report[case]=dict(**rates,within_nominal_bands=all(gates),expected_to_fail=expected_failure)
            print(case,report[case],flush=True)
    small=[]
    for n in range(2,13):
        d=rng.normal(size=n); exact=stats.paired(d,B,seed+n)['p'];null=stats.replicates(d,max(10000,B),seed+n)[2]
        estimate=stats.pvalue(null,float(d.mean()));se=math.sqrt(exact*(1-exact)/len(null))
        small.append(dict(n=n,exact=exact,monte_carlo=estimate,within_3se=abs(estimate-exact)<=3*se+1/(len(null)+1)))
    if file_hash(Path(stats.__file__))!=source_hash:raise ValueError('Statistics source changed during calibration')
    return dict(seed=seed,datasets=datasets,base_resamples=B,full_protocol=datasets>=1000 and B>=1000,
        statistics_sha256=source_hash,calibration_code_sha256=file_hash(Path(__file__)),workers=workers,
        passed=all(x['within_nominal_bands'] for x in report.values() if not x['expected_to_fail']) and all(x['within_3se'] for x in small),
        cases=report,T4e=small,interpretation='Empirical calibration for these cases only; marginal mean zero does not imply the joint sign-flip null.')
