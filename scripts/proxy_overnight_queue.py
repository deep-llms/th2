"""Fixed follow-up screen: fresh runs, explicit loss weights, safe predecessor gate."""
import argparse
import copy
import json
from pathlib import Path
import time

from deep_kv.__main__ import arm_jobs
from run_experiments import load_jobs
from scripts.proxy_12layer_queue import make as validate_recipe

# Labels identify runs; arm names identify architecture. Never alias a weight
# sweep into a different model class or overwrite the parent's output directory.
CASES = (
    ('p7-deep', 'P7-simple-deep', .1),
    ('p7-deep-shallow', 'P7-simple-deep-shallow', .1),
    ('consumer', 'P6-iso-consumer', .1),
    ('consumer-only', 'P6-iso-consumer-only', .1),
    ('p7-lm-low', 'P7-simple-lm', .03),
    ('p7-lm-high', 'P7-simple-lm', .3),
    ('p6-high', 'P6', .3),
    ('p6-higher', 'P6', 1.),
    ('targetgrad-low', 'P6-iso-targetgrad', .01),
    ('p7-sparse', 'P7-simple-sparse', .1),
    ('p7-short', 'P7-simple-short', .1),
)


def read(path):
    return json.loads(Path(path).read_text())


def predecessor_ready(root, guard=Path('/mnt/local/_gpu_guard/DISABLED')):
    """Require completed screen AND finished burn handoff. Never signals GPUs."""
    root=Path(root)/'supervised'
    path=root/'supervisor.json'
    if not path.exists():return False
    state=read(path)
    if state.get('training_status')=='failed':
        raise RuntimeError('Previous screen failed; follow-up remains unlaunched')
    if not state.get('finished_at'):return False
    burn=read(root/'burn-verified.json')
    run=read(root/'run/run.json')
    from scripts.proxy_12layer_queue import ALL_SCREEN_ARMS
    from deep_kv import P6_CONSUMER_ARMS
    required={f'seed-42-{kind}-{arm}' for arm in ALL_SCREEN_ARMS if arm not in P6_CONSUMER_ARMS
              for kind in ('arm','validate')}
    passed={j['name'] for j in run['jobs'] if j['status']=='ok'}
    if (state.get('training_status')!='ok' or state.get('training_returncode')!=0
            or run['status']!='ok' or not required.issubset(passed)
            or not state.get('burn',{}).get('collective_progress_verified')
            or not burn.get('collective_progress_verified')
            or burn.get('session')!=state['burn'].get('session')):
        raise RuntimeError('Previous screen or burn handoff did not validate')
    return not guard.exists()


def wait_for_predecessor(root):
    for attempt in range(1440):
        if predecessor_ready(root):
            print('PREDECESSOR_SCREEN_AND_BURN_VERIFIED',str(root),flush=True)
            return
        if attempt%10==0:print('WAITING_FOR_SCREEN_AND_BURN',attempt,flush=True)
        time.sleep(30)
    raise RuntimeError('Timed out waiting for previous screen; no GPU action taken')


def make(recipe_path, output, rows, predecessor=None):
    output=Path(output)
    # Reuse the existing fixed 12-layer recipe checks (without launching it).
    validate_recipe(recipe_path,output.with_name('recipe-validation-jobs.json'))
    base=read(recipe_path)
    selected=list(dict.fromkeys(arm for _,arm,_ in CASES))
    items=[dict(name='numerics',gpus=list(range(8)),argv=[
        '{python}','-u','-m','scripts.check_fa4_proxy','run','--recipe',str(recipe_path),
        '--rows',str(rows),'--worker-arms',*selected,'--output','{run_dir}/numerics'],
        required_outputs=[dict(path='numerics/summary.json',json_equals={'status':'passed'})])]
    production=[]
    for label,arm,weight in CASES:
        recipe={**base,'seed':42,'data_seed':42,'proxy_module_seed':43,'proxy_lambda_max':weight}
        job=arm_jobs(recipe,None,[arm])['jobs'][0]
        prefix=label+'/seed-42/'
        job['name']=label+'-train'
        job['argv']=['env','PYTHONHASHSEED=42',*[a.replace('{run_dir}','{run_dir}/'+label+'/seed-42') for a in job['argv']]]
        job['argv'][job['argv'].index('--run_name')+1]='screen-followup-'+label
        for r in job['required_outputs']:r['path']=prefix+r['path']
        validator=dict(name=label+'-validate',argv=[
            '{python}','-m','scripts.check_fa4_proxy','validate','--run-dir','{run_dir}/'+label,
            '--steps','2500','--seed','42','--schedule-steps','2500','--warmup-steps','125',
            '--num-hidden-layers','12','--arms',arm,'--attention-backend','fa4',
            '--expected-lambda',str(weight),'--output','{run_dir}/'+label+'/validated.json'],
            required_outputs=[dict(path=label+'/validated.json',json_equals={'status':'passed'})])
        production += [job,validator]
        smoke=copy.deepcopy(job);smoke['name']='smoke-'+smoke['name']
        smoke['argv']=[a.replace('{run_dir}','{run_dir}/smoke') for a in smoke['argv']]
        for key,value in dict(stop_after=3,save_steps=3,logging_steps=1,eval_steps=3,
                              eval_rows=16,monitor_rows=16,eval_on_start=False).items():
            smoke['argv'][smoke['argv'].index('--'+key)+1]=str(value).lower()
        for r in smoke['required_outputs']:
            r['path']='smoke/'+r['path'];r['json_equals'].update(global_step=3,status='stopped')
        check=copy.deepcopy(validator);check['name']='smoke-'+check['name']
        check['argv']=[a.replace('{run_dir}','{run_dir}/smoke') for a in check['argv']]
        check['argv'][check['argv'].index('--steps')+1]='3'
        for r in check['required_outputs']:r['path']='smoke/'+r['path']
        items += [smoke,check]
    items += production
    items.append(dict(name='summarize',argv=['{python}','-m','scripts.proxy_overnight_queue',
        'summarize','--root','{run_dir}',*(['--baseline',str(Path(predecessor)/'supervised/run/seed-42/A')] if predecessor else [])],required_outputs=[dict(path='summary.json',json_equals={'status':'passed'})]))
    with output.open('x') as f:json.dump(dict(jobs=items),f,indent=2)
    load_jobs(output)
    return items


def summarize(root, baseline=None):
    root=Path(root);results=[];fingerprints=None
    for label,arm,weight in CASES:
        folder=root/label/'seed-42'/arm
        r=read(folder/'result.json');c=read(folder/'train_config.json')
        assert read(root/label/'validated.json')['status']=='passed'
        assert r['global_step']==2500 and r['arm']==arm and r['status']=='complete'
        assert c['pilot']['proxy_lambda_max']==weight
        pair=(c['train_fingerprint'],c['eval_fingerprint'])
        if fingerprints is None:fingerprints=pair
        assert pair==fingerprints,'Follow-up runs used different data'
        results.append(dict(label=label,arm=arm,lambda_max=weight,evaluation=r['evaluation'],training_cost=r['training_cost']))
    reference=None
    if baseline:
        baseline=Path(baseline);a=read(baseline/'result.json');c=read(baseline/'train_config.json')
        assert a['arm']=='A' and a['global_step']==2500 and a['status']=='complete'
        assert c['training']['seed']==c['training']['data_seed']==42
        assert (c['train_fingerprint'],c['eval_fingerprint'])==fingerprints
        reference=dict(path=str(baseline),evaluation=a['evaluation'])
        for r in results:
            assert r['evaluation']['eval_target_tokens']==a['evaluation']['eval_target_tokens']
            r['lm_loss_delta_vs_A']=r['evaluation']['eval_lm_loss']-a['evaluation']['eval_lm_loss']
    with (root/'summary.json').open('x') as f:
        json.dump(dict(status='passed',results=results,baseline=reference,train_eval_fingerprints=fingerprints),f,indent=2,allow_nan=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('summarize',));p.add_argument('--root',required=True)
    p.add_argument('--baseline');args=p.parse_args();summarize(args.root,args.baseline)
