"""Prepare P7 tuning jobs for the existing supervisor; never launch GPU work."""
import argparse
import copy
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


def make(recipe_path, output, rows, labels=None):
    """All numerical/distributed smokes precede training, in separate outputs."""
    from scripts.proxy_12layer_queue import make as validate_recipe
    base=json.loads(Path(recipe_path).read_text())
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    cases=CASES if labels is None else tuple(c for c in CASES if c[0] in labels)
    if labels is not None and (not labels or len(set(labels))!=len(labels) or len(cases)!=len(labels)):
        raise ValueError('Select distinct known case labels')
    config_root=output.with_suffix('.config');config_root.mkdir(parents=True,exist_ok=False)
    validate_recipe(recipe_path,config_root/'recipe-validation.json')
    items=[];production=[]
    numerical_arms=list(dict.fromkeys(c[1] for c in cases if c[1]!='A'))
    if numerical_arms:
        items.append(dict(name='numerics',gpus=list(range(8)),timeout_seconds=7200,argv=[
            '{python}','-u','-m','scripts.check_fa4_proxy','run','--recipe',str(Path(recipe_path).resolve()),
            '--rows',str(rows),'--worker-arms',*numerical_arms,'--output','{run_dir}/numerics'],
            required_outputs=[dict(path='numerics/summary.json',json_equals={'status':'passed'})]))
    for label,arm,weight,seed in cases:
        recipe={**base,**(settings_overrides(arm) if arm in TUNING_ARMS else {}),
            'max_steps':2500,'stop_after':2500,'warmup_steps':125,
            'seed':seed,'data_seed':seed,'proxy_module_seed':seed+1 if arm!='A' else None,
            'proxy_lambda_max':weight,'save_steps':250,'save_total_limit':0}
        # Keep new mathematical variants off the fixed-four-block custom kernel.
        for key in ('compile_auxiliary','compile_entries','lse_attention','fused_targets'):
            recipe['proxy_'+key]=arm=='P7-simple-targetgrad'
        if arm in NOAUX_ARMS:assert weight==0
        (config_root/(label+'.json')).write_text(json.dumps(recipe,indent=2)+'\n')
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
        for key,value in dict(stop_after=3,save_steps=3,logging_steps=1,eval_steps=3,
                              eval_rows=16,monitor_rows=16,eval_on_start=False).items():
            smoke['argv'][smoke['argv'].index('--'+key)+1]=str(value).lower()
        for r in smoke['required_outputs']:
            r['path']='smoke/'+r['path'];r['json_equals'].update(global_step=3,status='stopped')
        smoke_check=copy.deepcopy(check);smoke_check['name']='smoke-'+check['name']
        smoke_check['argv']=[a.replace('{run_dir}','{run_dir}/smoke') for a in smoke_check['argv']]
        smoke_check['argv'][smoke_check['argv'].index('--steps')+1]='3'
        for r in smoke_check['required_outputs']:r['path']='smoke/'+r['path']
        items.extend((smoke,smoke_check));production.extend((job,check))
    items.extend(production)
    with output.open('x') as f:json.dump(dict(jobs=items),f,indent=2)
    load_jobs(output)
    return items


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',required=True);p.add_argument('--output',required=True)
    p.add_argument('--rows',required=True)
    p.add_argument('--cases',nargs='+',choices=[c[0] for c in CASES])
    a=p.parse_args();make(a.recipe,a.output,a.rows,a.cases)
