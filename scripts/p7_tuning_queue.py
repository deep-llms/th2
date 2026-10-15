"""Prepare P7 tuning jobs for the existing supervisor; never launch GPU work."""
import argparse
import copy
from dataclasses import asdict, fields
import json
from pathlib import Path

from deep_kv.__main__ import arm_jobs
from deep_kv.p7_variants import TUNING_ARMS, NOAUX_ARMS, settings_overrides
from run_experiments import load_jobs


# label, arm, lambda, training/data seed. Module seed is training seed + 1.
CASES = (
    ('tg-lambda003', 'P7-simple-targetgrad', .03, 42),
    ('tg-lambda03', 'P7-simple-targetgrad', .3, 42),
    ('tg-short', 'P7-simple-targetgrad-short', .1, 42),
    ('tg-sparse', 'P7-simple-targetgrad-sparse', .1, 42),
    ('tg-heads2', 'P7-simple-targetgrad-heads2', .1, 42),
    ('tg-heads8', 'P7-simple-targetgrad-heads8', .1, 42),
    ('tg-keyonly', 'P7-simple-targetgrad-keyonly', .1, 42),
    ('tg-valueonly', 'P7-simple-targetgrad-valueonly', .1, 42),
    ('lm-lambda001', 'P7-simple-lm', .01, 42),
    ('lm-lambda1', 'P7-simple-lm', 1., 42),
    ('lm-sparse-noaux', 'P7-simple-lm-sparse-noaux', 0., 42),
    ('lm-keyonly-noaux', 'P7-simple-lm-keyonly-noaux', 0., 42),
    ('lm-valueonly-noaux', 'P7-simple-lm-valueonly-noaux', 0., 42),
    ('lm-valueonly-gated', 'P7-simple-lm-valueonly-gated', .1, 42),
    ('a-seed2042', 'A', .1, 2042),
    ('lm-noaux-seed2042', 'P7-simple-lm-noaux', 0., 2042),
    ('tg-medium', 'P7-simple-targetgrad-medium', .1, 42),
    ('tg-weighted', 'P7-simple-targetgrad-weighted', .1, 42),
    ('tg-huber', 'P7-simple-targetgrad-huber', .1, 42),
    ('tg-nonlinear', 'P7-simple-targetgrad-nonlinear', .1, 42),
)
LEFTOVER_CASES = (
    ('lm-noaux', 'P7-simple-lm-noaux', 0., 42),
    ('lm-nexttoken', 'P7-simple-lm-nexttoken', .1, 42),
)
ALL_CASES = LEFTOVER_CASES + CASES
SMOKE_OVERRIDES = dict(stop_after=3,save_steps=3,logging_steps=1,eval_steps=3,
                      eval_rows=16,monitor_rows=16,eval_on_start=False)


def check_case_config(config, planned, smoke=False):
    """Check the resolved run against its own requested recipe, not another arm."""
    from train import ModelArguments, DataArguments, PilotArguments
    from transformers import TrainingArguments
    from deep_kv.proxy import ProxySettings
    from scripts.check_fa4_proxy import numerical_settings
    requested={**planned,**(SMOKE_OVERRIDES if smoke else {})}
    actual={**config['training'],**config['pilot']}
    expected_settings,_=numerical_settings(planned['arm'],planned,exact=True)
    actual_settings,_=numerical_settings(planned['arm'],actual,exact=True)
    if asdict(expected_settings)!=asdict(actual_settings):
        raise ValueError('Resolved proxy settings differ from the requested case')
    sections=((ModelArguments,'model'),(DataArguments,'data'),(PilotArguments,'pilot'),(TrainingArguments,'training'))
    mapping={f.name:section for cls,section in sections for f in fields(cls)}
    defaults=asdict(PilotArguments())
    proxy_fields={'proxy_'+key for key in ProxySettings.__dataclass_fields__}
    for key,value in requested.items():
        if key in ('output_dir','run_name','stop_after') or value is None or key in proxy_fields:continue
        if key not in mapping:raise ValueError('Unrecognized recipe field: '+key)
        observed=config[mapping[key]].get(key,defaults.get(key))
        if key=='report_to' and isinstance(value,str):value=[] if value=='none' else [value]
        if observed!=value:raise ValueError('Run differs from requested recipe field: '+key)


def verify(root, config_root, labels, smoke=False):
    """Gate same-seed data order, common recipes and checkpoint retention."""
    from deep_kv.report import comparable_config, report
    root=Path(root);config_root=Path(config_root)
    cases=tuple(c for c in ALL_CASES if c[0] in labels)
    if not labels or len(cases)!=len(labels) or len(set(labels))!=len(labels):
        raise ValueError('Select distinct known case labels')
    phase_root=root/'smoke' if smoke else root
    common=None;fingerprints={};rows=[]
    for label,arm,weight,seed in cases:
        folder=phase_root/label/f'seed-{seed}'/arm
        read=lambda p:json.loads(Path(p).read_text())
        config=read(folder/'train_config.json');planned=read(config_root/(label+'.json'))
        if (planned['arm']!=arm or planned['seed']!=seed or planned['data_seed']!=seed
                or planned['proxy_lambda_max']!=weight
                or (arm!='A' and planned['proxy_module_seed']!=seed+1)):
            raise ValueError('Case recipe does not match the fixed experiment: '+label)
        if read(phase_root/label/'validated.json')['status']!='passed':
            raise ValueError('Missing successful case validation: '+label)
        check_case_config(config,planned,smoke)
        step=3 if smoke else 2500
        report(folder.parent,[arm],expected_step=step,expected_seed=seed,expected_attention_backend='fa4')
        checkpoints={int(p.name.removeprefix('checkpoint-')) for p in folder.glob('checkpoint-*') if p.is_dir()}
        if checkpoints!=({3} if smoke else set(range(250,2501,250))):
            raise ValueError('Checkpoint retention mismatch: '+label)
        # A different data seed intentionally changes the shuffled train order.
        pair=(config['train_fingerprint'],config['eval_fingerprint'])
        if seed in fingerprints and pair!=fingerprints[seed]:
            raise ValueError('Same-seed data fingerprints differ: '+label)
        fingerprints[seed]=pair
        normalized=comparable_config(config)
        normalized.pop('train_fingerprint')
        normalized['training'].update(seed=42,data_seed=42)
        normalized['pilot']['proxy_lambda_max']=.1  # Checked against its case above.
        if common is not None and normalized!=common:
            raise ValueError('Shared recipe or evaluation fingerprint differs: '+label)
        common=normalized
        result=read(folder/'result.json')
        rows.append(dict(label=label,arm=arm,seed=seed,lambda_max=weight,step=step,
                         evaluation=result['evaluation'],training_cost=result.get('training_cost')))
    result=dict(status='passed',data_fingerprints=fingerprints,cases=rows)
    with (root/('smoke-recipes.json' if smoke else 'summary.json')).open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)
    return result


def make(recipe_path, output, rows, labels=None):
    """All numerical/distributed smokes precede training, in separate outputs."""
    from scripts.proxy_12layer_queue import make as validate_recipe
    base=json.loads(Path(recipe_path).read_text())
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    cases=CASES if labels is None else tuple(c for c in ALL_CASES if c[0] in labels)
    if labels is not None and (not labels or len(set(labels))!=len(labels) or len(cases)!=len(labels)):
        raise ValueError('Select distinct known case labels')
    config_root=output.with_suffix('.config');config_root.mkdir(parents=True,exist_ok=False)
    validate_recipe(recipe_path,config_root/'recipe-validation.json')
    items=[];production=[];numerical_cases=[]
    for label,arm,weight,seed in cases:
        recipe={**base,**(settings_overrides(arm) if arm in TUNING_ARMS else {}),
            'arm':arm,'max_steps':2500,'stop_after':2500,'warmup_steps':125,
            'seed':seed,'data_seed':seed,'proxy_module_seed':seed+1 if arm!='A' else None,
            'proxy_lambda_max':weight,'save_steps':250,'save_total_limit':0}
        # Keep new mathematical variants off the fixed-four-block custom kernel.
        for key in ('compile_auxiliary','compile_entries','lse_attention','fused_targets'):
            recipe['proxy_'+key]=arm=='P7-simple-targetgrad'
        if arm in NOAUX_ARMS:assert weight==0
        (config_root/(label+'.json')).write_text(json.dumps(recipe,indent=2)+'\n')
        if arm!='A':numerical_cases.append(dict(label=label,arm=arm,recipe=str((config_root/(label+'.json')).resolve())))
        job=arm_jobs(recipe,None,[arm])['jobs'][0]
        prefix=f'{label}/seed-{seed}'
        job.update(name=label+'-train',timeout_seconds=21600)
        job['argv']=['env',f'PYTHONHASHSEED={seed}',*[a.replace('{run_dir}','{run_dir}/'+prefix) for a in job['argv']]]
        job['argv'][job['argv'].index('--run_name')+1]='p7-tuning-'+label
        for r in job['required_outputs']:r['path']=prefix+'/'+r['path']
        check=dict(name=label+'-validate',argv=[
            '{python}','-m','scripts.check_fa4_proxy','validate','--run-dir','{run_dir}/'+label,
            '--steps','2500','--seed',str(seed),'--schedule-steps','2500','--warmup-steps','125',
            '--num-hidden-layers','12','--arms',arm,'--attention-backend','fa4',
            '--expected-lambda',str(weight),'--output','{run_dir}/'+label+'/validated.json'],
            required_outputs=[dict(path=label+'/validated.json',json_equals={'status':'passed'})])
        smoke=copy.deepcopy(job);smoke['name']='smoke-'+job['name'];smoke['timeout_seconds']=1800
        smoke['argv']=[a.replace('{run_dir}','{run_dir}/smoke') for a in smoke['argv']]
        for key,value in SMOKE_OVERRIDES.items():
            smoke['argv'][smoke['argv'].index('--'+key)+1]=str(value).lower()
        for r in smoke['required_outputs']:
            r['path']='smoke/'+r['path'];r['json_equals'].update(global_step=3,status='stopped')
        smoke_check=copy.deepcopy(check);smoke_check['name']='smoke-'+check['name']
        smoke_check['argv']=[a.replace('{run_dir}','{run_dir}/smoke') for a in smoke_check['argv']]
        smoke_check['argv'][smoke_check['argv'].index('--steps')+1]='3'
        for r in smoke_check['required_outputs']:r['path']='smoke/'+r['path']
        items.extend((smoke,smoke_check));production.extend((job,check))
    if numerical_cases:
        manifest=config_root/'numerical-cases.json';manifest.write_text(json.dumps(numerical_cases,indent=2)+'\n')
        items.insert(0,dict(name='numerics',gpus=list(range(8)),timeout_seconds=7200,argv=[
            '{python}','-u','-m','scripts.check_fa4_proxy','run','--case-manifest',str(manifest.resolve()),
            '--rows',str(rows),'--output','{run_dir}/numerics'],
            required_outputs=[dict(path='numerics/summary.json',json_equals={'status':'passed'})]))
    def verification_job(smoke):
        name='smoke-recipes' if smoke else 'summary'
        return dict(name=name,argv=['{python}','-m','scripts.p7_tuning_queue','--verify-root','{run_dir}',
            '--config-root',str(config_root.resolve()),'--cases',*[c[0] for c in cases],*(['--smoke'] if smoke else [])],
            required_outputs=[dict(path=name+'.json',json_equals={'status':'passed'})])
    items.append(verification_job(True))
    items.extend(production)
    items.append(verification_job(False))
    with output.open('x') as f:json.dump(dict(jobs=items),f,indent=2)
    load_jobs(output)
    return items


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe');p.add_argument('--output');p.add_argument('--rows')
    p.add_argument('--verify-root');p.add_argument('--config-root');p.add_argument('--smoke',action='store_true')
    p.add_argument('--cases',nargs='+',choices=[c[0] for c in ALL_CASES])
    a=p.parse_args()
    if a.verify_root:
        if not a.config_root or not a.cases:p.error('Verification requires --config-root and --cases')
        verify(a.verify_root,a.config_root,a.cases,a.smoke)
    else:
        if not a.recipe or not a.output or not a.rows:p.error('Generation requires --recipe, --output and --rows')
        make(a.recipe,a.output,a.rows,a.cases)
