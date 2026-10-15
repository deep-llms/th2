"""Prepare the bounded P7 exploration queue; never launches or signals GPUs.

All runs are fresh. The old 2500-step schedules have already completed; changing
their horizon is not an ordinary optimizer/scheduler/data-order resume.
"""
import argparse
import copy
import json
from pathlib import Path

from deep_kv import P7_EXPLORATION_ARMS
from deep_kv.__main__ import arm_jobs
from scripts.proxy_overnight_queue import read
from scripts.proxy_confirmation_queue import EVAL_PYTHON, evaluation_reference
from scripts.p7_speed_study import write as save
from run_experiments import load_jobs


CASES = (
    ('a-long', 'A', 7500, .1),
    ('targetgrad-long', 'P7-simple-targetgrad', 7500, .1),
    ('lm-long', 'P7-simple-lm', 7500, .1),
    ('lm-sparse', 'P7-simple-lm-sparse', 2500, .1),
    ('lm-keyonly', 'P7-simple-lm-keyonly', 2500, .1),
    ('lm-valueonly', 'P7-simple-lm-valueonly', 2500, .1),
    ('lm-noaux', 'P7-simple-lm-noaux', 2500, 0.),
    ('lm-nexttoken', 'P7-simple-lm-nexttoken', 2500, .1),
)
OPTIMIZATIONS = ('proxy_compile_auxiliary','proxy_lse_attention','proxy_compile_entries','proxy_fused_targets')


def checkpoint_path(root, case):
    label, arm, steps, _ = case
    return Path(root)/label/'seed-42'/arm/f'checkpoint-{steps}'


def make(recipe_path, output, rows, reference):
    from scripts.proxy_12layer_queue import make as validate_recipe
    output=Path(output);config_root=output.with_suffix('.config')
    if output.exists():raise FileExistsError(output)
    config_root.mkdir(parents=True,exist_ok=False)
    validate_recipe(recipe_path,config_root/'base-validation.json')
    base=read(recipe_path)
    if base.get('seed')!=42 or base.get('data_seed')!=42:raise ValueError('This screen fixes seed 42')
    if any(base.get(k,False) for k in OPTIMIZATIONS):raise ValueError('Set optimizations per arm, not globally')
    selected=['P7-simple-targetgrad','P7-simple-lm',*P7_EXPLORATION_ARMS]
    items=[dict(name='numerics',gpus=list(range(8)),timeout_seconds=3600,argv=[
        '{python}','-u','-m','scripts.check_fa4_proxy','run','--recipe',str(Path(recipe_path).resolve()),
        '--rows',str(rows),'--worker-arms',*selected,'--output','{run_dir}/numerics'],
        required_outputs=[dict(path='numerics/summary.json',json_equals={'status':'passed'})])]
    production=[]
    for case in CASES:
        label,arm,steps,weight=case
        recipe={**base,'max_steps':steps,'stop_after':steps,'warmup_steps':steps//20,
                'proxy_lambda_max':weight,'proxy_module_seed':43 if arm!='A' else None}
        if arm=='P7-simple-targetgrad':recipe.update({k:True for k in OPTIMIZATIONS})
        save(config_root/(label+'.json'),recipe)
        job=arm_jobs(recipe,None,[arm])['jobs'][0]
        prefix=label+'/seed-42'
        job.update(name=label+'-train',timeout_seconds=21600)
        job['argv']=['env','PYTHONHASHSEED=42',*[a.replace('{run_dir}','{run_dir}/'+prefix) for a in job['argv']]]
        job['argv'][job['argv'].index('--run_name')+1]='p7-exploration-'+label
        for r in job['required_outputs']:r['path']=prefix+'/'+r['path']
        check=dict(name=label+'-validate',argv=[
            '{python}','-m','scripts.check_fa4_proxy','validate','--run-dir','{run_dir}/'+label,
            '--steps',str(steps),'--seed','42','--schedule-steps',str(steps),
            '--warmup-steps',str(steps//20),'--num-hidden-layers','12','--arms',arm,
            '--attention-backend','fa4','--expected-lambda',str(weight),
            '--output','{run_dir}/'+label+'/validated.json'],
            required_outputs=[dict(path=label+'/validated.json',json_equals={'status':'passed'})])
        production.extend((job,check))
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
        items.extend((smoke,smoke_check))
    items.append(dict(name='verify-smoke-recipes',argv=['{python}','-m','scripts.p7_exploration_queue',
        'verify-smokes','--root','{run_dir}','--reference',str(reference)],
        required_outputs=[dict(path='smoke-recipes.json',json_equals={'status':'passed'})]))
    items+=production
    items.append(dict(name='summarize',argv=['{python}','-m',__name__,'summarize',
        '--root','{run_dir}','--reference',str(reference)],
        required_outputs=[dict(path='summary.json',json_equals={'status':'passed'})]))
    for stage in ('numerics','smoke','full'):
        items.append(dict(name='eval-'+stage,gpus=list(range(8)),timeout_seconds=14400,
            argv=[EVAL_PYTHON,'-u','-m','scripts.p7_exploration_queue','evaluate',
                  '--root','{run_dir}','--reference',str(reference),'--stage',stage],
            required_outputs=[dict(path=f'downstream/{stage}/summary.json',json_equals={'status':'passed'})]))
    # Explicit module name also when this file was invoked with python -m.
    for job in items:
        job['argv']=['scripts.p7_exploration_queue' if a=='__main__' else a for a in job['argv']]
    save(output,dict(jobs=items));load_jobs(output)
    return items


def verify_smokes(root, reference):
    """Reject changed training data/recipes before spending on full runs."""
    from deep_kv.report import comparable_config
    original=Path(read(Path(reference)/'scope.json')['source_screen'])/'supervised/run/seed-42'
    reference_config=read(original/'A/train_config.json')
    reference_result=read(original/'A/result.json')
    if reference_result['global_step']!=2500 or reference_result['status']!='complete':
        raise ValueError('Original A is not complete')
    def normalized(config):
        c=comparable_config(config)
        # Smokes deliberately reduce logging/evaluation/save intervals and
        # validation prefix size. Their full TRAIN fingerprint must still match.
        for key in ('logging_steps','save_steps','eval_steps','eval_on_start'):
            c['training'].pop(key,None)
        for key in ('eval_rows','monitor_rows'):c['data'].pop(key,None)
        c.pop('eval_fingerprint',None)
        c['training']['max_steps']=2500;c['training']['warmup_steps']=125
        return c
    expected=normalized(reference_config)
    for case in CASES:
        label,arm,steps,_=case;folder=checkpoint_path(Path(root)/'smoke',case).parent
        config=read(folder/'train_config.json')
        if (read(Path(root)/'smoke'/label/'validated.json')['status']!='passed'
                or config['training']['max_steps']!=steps or config['training']['warmup_steps']!=steps//20
                or config['pilot']['arm']!=arm or normalized(config)!=expected):
            raise ValueError('Smoke recipe or training data differs from the control: '+label)
    save(Path(root)/'smoke-recipes.json',dict(status='passed',cases=len(CASES),
         train_fingerprint=reference_config['train_fingerprint'],reference=str(original/'A')))


def validate_training(root, cases=CASES):
    from deep_kv.report import report
    rows=[];fingerprints=None
    for case in cases:
        label,arm,steps,weight=case;ck=checkpoint_path(root,case);folder=ck.parent
        c=read(folder/'train_config.json');r=read(folder/'result.json')
        if (read(Path(root)/label/'validated.json')['status']!='passed'
                or r['arm']!=arm or r['global_step']!=steps or r['status']!='complete'
                or read(ck/'trainer_state.json')['global_step']!=steps
                or c['training']['max_steps']!=steps or c['training']['warmup_steps']!=steps//20
                or c['training']['seed']!=42 or c['training']['data_seed']!=42
                or c['training']['save_total_limit']!=0 or c['training']['save_steps']!=250
                or c['model_config']['num_hidden_layers']!=12
                or c['pilot']['arm']!=arm or c['pilot']['proxy_lambda_max']!=weight
                or c['pilot']['attention_backend']!='fa4'
                or (arm!='A' and c['pilot']['proxy_module_seed']!=43)
                or any(c['pilot'].get(k,False)!=(arm=='P7-simple-targetgrad') for k in OPTIMIZATIONS)):
            raise ValueError('Exploration checkpoint/recipe mismatch: '+label)
        # Validate scientific metadata, metrics and schedule through the shared report.
        report(folder.parent,[arm],expected_step=steps,expected_seed=42,expected_attention_backend='fa4')
        pair=(c['train_fingerprint'],c['eval_fingerprint'])
        if fingerprints is None:fingerprints=pair
        if pair!=fingerprints:raise ValueError('Exploration data fingerprints differ')
        rows.append(dict(label=label,arm=arm,step=steps,lambda_max=weight,seed=42,
            checkpoint=str(ck),evaluation=r['evaluation'],training_cost=r['training_cost']))
    return rows


def summarize(root, reference):
    from deep_kv.report import comparable_config
    rows=validate_training(root)
    previous=read(Path(reference)/'scope.json')
    original=Path(previous['source_screen'])/'supervised/run/seed-42'
    references={}
    for arm in ('A','P7-simple-lm'):
        folder=original/arm;r=read(folder/'result.json');c=read(folder/'train_config.json')
        if (r['arm']!=arm or r['global_step']!=2500 or r['status']!='complete'
                or c['training']['max_steps']!=2500 or c['training']['warmup_steps']!=125
                or c['training']['seed']!=42 or c['training']['data_seed']!=42):
            raise ValueError('Invalid short-schedule reference '+arm)
        references[arm]=dict(path=str(folder),evaluation=r['evaluation'])
        for case in CASES[3:]:
            candidate=read(checkpoint_path(root,case).parent/'train_config.json')
            if comparable_config(c)!=comparable_config(candidate):
                raise ValueError('Short-schedule reference recipe/data differs: '+arm+'/'+case[0])
    long_base=rows[0]['evaluation']
    for row in rows:
        baseline=long_base if row['step']==7500 else references['A']['evaluation']
        if row['evaluation']['eval_target_tokens']!=baseline['eval_target_tokens']:
            raise ValueError('Evaluation coverage differs')
        row['lm_loss_delta_vs_matched_A']=row['evaluation']['eval_lm_loss']-baseline['eval_lm_loss']
        row['comparison_schedule']=row['step']
        if row['step']==2500:
            row['lm_loss_delta_vs_short_lm']=row['evaluation']['eval_lm_loss']-references['P7-simple-lm']['evaluation']['eval_lm_loss']
    for row in rows[1:3]:
        a=comparable_config(read(checkpoint_path(root,CASES[0]).parent/'train_config.json'))
        b=comparable_config(read(Path(row['checkpoint']).parent/'train_config.json'))
        if a!=b:raise ValueError('Long-run recipes differ')
    save(Path(root)/'summary.json',dict(status='passed',results=rows,short_references=references,
        note='Fresh full schedules; do not compare long-schedule intermediate checkpoints to short-schedule endpoints'))


def validate_scope(scope):
    expected=[str(checkpoint_path(scope['training_root'],case)) for case in CASES]
    if scope['checkpoints']!=expected:raise ValueError('Exploration checkpoint set changed')
    for ck,(label,arm,steps,weight) in zip(expected,CASES):
        spec=scope['specs'][ck]
        if (spec['label'],spec['arm'],spec['step'],spec['lambda_max'],spec['seed'])!=(label,arm,steps,weight,42):
            raise ValueError('Exploration checkpoint identity changed')


def evaluate(root, reference, stage):
    from scripts.proxy_screen_evaluation import run
    from scripts.train_then_burn import digest
    root=Path(root);path=root/'downstream-scope.json'
    if stage=='numerics':
        previous=evaluation_reference(reference)
        if read(root/'summary.json')['status']!='passed':raise ValueError('Training not validated')
        rows=validate_training(root);specs={}
        for row in rows:
            specs[row['checkpoint']]={k:row[k] for k in ('label','arm','step','lambda_max','seed')}
            specs[row['checkpoint']]['checkpoint_sha256']=digest(Path(row['checkpoint'])/'model.safetensors')
        scope=dict(kind='p7-exploration8',training_root=str(root),checkpoints=list(specs),specs=specs,
            protocols=previous['protocols'],data_root=previous['data_root'],manifest=previous['manifest'],
            manifest_sha256=previous['manifest_sha256'],versions=previous['versions'],
            evaluation_seed=42,reference=str(reference),finetuning=False,paired_loss=False)
        validate_scope(scope);save(path,scope)
    run(stage,root/'downstream'/stage,path)


def prepare(root, host, predecessor, reference, rows):
    """Future remote preflight only. Supervisor owns all process signals."""
    import importlib.metadata as md
    import shutil
    import socket
    import subprocess
    import sys
    from accelerate.commands.config.config_args import default_yaml_config_file,load_config_from_file
    from scripts.p7_speed_study import predecessor_ready
    from scripts.train_then_burn import digest,approved_launcher,APPROVED_BURNS,BURN_HASH,GUARD_HASH
    from scripts.verified_gpu_reclaim import inspect,process
    root=Path(root)
    if socket.gethostname()!=host:raise ValueError('Wrong machine')
    predecessor_ready(predecessor)  # Completion AND verified burn handoff, before inspection/signals.
    expected=dict(torch='2.14.1',transformers='5.9.0',accelerate='1.13.0',datasets='4.8.5',
                  **{'flash-attn-4':'4.0.0b33','nvidia-cutlass-dsl':'4.8.0'})
    versions={k:md.version(k) for k in expected}
    if any(versions[k].split('+')[0]!=v for k,v in expected.items()):raise ValueError(versions)
    if root.exists():raise FileExistsError(root)
    if shutil.disk_usage(root.parent).free<2000*2**30:raise ValueError('Need 2000 GiB free for all retained checkpoints')
    recipe=read('proxy_heads.12layer.b200.json')
    for key in ('config_name','tokenizer_name','data_dir','eval_data_dir'):
        if not Path(recipe[key]).exists():raise ValueError('Missing local '+key)
    if not Path(rows).is_file():raise ValueError('Missing packed numerical rows')
    from scripts.check_fa4_proxy import diagnostic_rows
    diagnostic_rows(rows)  # Validate the actual format/boundaries before reclaiming GPUs.
    # This command checks the pinned evaluation environment and local benchmark
    # files; it neither allocates GPUs nor downloads anything.
    root.mkdir()
    subprocess.run([EVAL_PYTHON,'-m','scripts.proxy_confirmation_queue','preflight',
                    '--root',str(root),'--reference',str(reference)],check=True)
    source=Path('resources/accelerate_config.yaml');destination=Path(default_yaml_config_file)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():shutil.copy2(destination,root/'accelerate.previous.yaml')
    shutil.copy2(source,destination);cfg=load_config_from_file(str(destination)).to_dict()
    if (source.read_bytes()!=destination.read_bytes() or cfg['num_processes']!=8
            or cfg['mixed_precision']!='bf16' or cfg['distributed_type']!='MULTI_GPU'):
        raise ValueError('Accelerate mismatch')
    subprocess.run([str(Path(sys.executable).parent/'accelerate'),'env'],check=True)
    make('proxy_heads.12layer.b200.json',root/'jobs.json',rows,reference)
    status=inspect()
    if (status['host']!=host or status['guard_disabled'] or len(status['gpus'])!=8
            or any('B200' not in g['name'] for g in status['gpus'])
            or digest('/mnt/local/_gpu_guard/gpu_guard.sh')!=GUARD_HASH):
        raise ValueError('GPU ownership/state changed')
    approved={**APPROVED_BURNS,str(Path.cwd()/'resources/llm_pretrain_burn.py'):BURN_HASH}
    if any(not (approved_launcher(pid,approved) or approved_launcher(process(pid)['ppid'],approved))
           for pid in status['workers']):raise ValueError('Unknown GPU workload; no signals sent')
    save(root/'inspection.json',status)
    save(root/'preflight.json',dict(status='passed',versions=versions,predecessor=str(predecessor),
        reference=str(reference),rows_sha256=digest(rows),accelerate_sha256=digest(destination),
        environment=sys.executable,source_sha256={p:digest(p) for p in (
            'train.py','deep_kv/proxy.py','deep_kv/proxy_memory.py','deep_kv/proxy_training.py',
            'scripts/p7_exploration_queue.py','scripts/check_fa4_proxy.py')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('make','summarize','evaluate','prepare','verify-smokes'))
    p.add_argument('--recipe',default='proxy_heads.12layer.b200.json');p.add_argument('--output')
    p.add_argument('--rows');p.add_argument('--reference',required=True);p.add_argument('--root')
    p.add_argument('--stage',choices=('numerics','smoke','full'))
    p.add_argument('--host');p.add_argument('--predecessor')
    a=p.parse_args()
    if a.mode=='make':
        if not a.output or not a.rows:p.error('make requires --output and --rows')
        make(a.recipe,a.output,a.rows,a.reference)
    elif not a.root:p.error('summary/evaluation requires --root')
    elif a.mode=='prepare':
        if not all((a.host,a.predecessor,a.rows)):p.error('prepare requires --host, --predecessor and --rows')
        prepare(a.root,a.host,a.predecessor,a.reference,a.rows)
    elif a.mode=='summarize':summarize(a.root,a.reference)
    elif a.mode=='verify-smokes':verify_smokes(a.root,a.reference)
    elif not a.stage:p.error('evaluate requires --stage')
    else:evaluate(a.root,a.reference,a.stage)
