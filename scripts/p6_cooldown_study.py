"""Three preserved-checkpoint cooldown branches and full held-out PPL comparison."""
import argparse
import json
import math
from pathlib import Path
import sys

from deep_kv.cooldown import stage
from scripts.p6_compile_trial import read, write
from scripts.stage_deep_kv_resume import digest


def duration_for_budget(p6_total, a_prior, a_seconds_per_step):
    if not all(math.isfinite(x) and x>0 for x in (p6_total,a_prior,a_seconds_per_step)):
        raise ValueError('Invalid measured runtime')
    steps=round((p6_total-a_prior)/a_seconds_per_step)
    if not 1<=steps<=28600-10800:raise ValueError('Invalid time-matched cooldown budget')
    return steps


def branch(root, name, checkpoint, steps, optimized):
    root=Path(root);checkpoint=Path(checkpoint)
    config=read(checkpoint.parent/'train_config.json')
    start=read(checkpoint/'trainer_state.json')['global_step']
    output=root/name
    receipt=stage(checkpoint,output,start,steps)
    args={**config['model'],**config['data'],**config['pilot'],**config['training'],
          'output_dir':str(output),'resume_from_checkpoint':receipt['checkpoint'],
          'cooldown_start':start,'cooldown_steps':steps,'stop_after':start+steps,
          'save_total_limit':0,'proxy_sync_optimization':optimized,'allow_performance_change_on_resume':True}
    write(root/(name+'.json'),args)


def prepare(root, benchmark):
    from deep_kv.report import comparable_config
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    benchmark=Path(benchmark)
    result=read(benchmark/'summary.json');gate=read(benchmark/'audit/verified.json')
    if result['status']!='passed' or gate['status']!='passed' or not gate['exact']:
        raise ValueError('Exact-state and timing gates must pass first')
    means=result['mean_seconds']
    optimized=(means['P6-optimized']<means['P6-reference']*.998
               and means['A-optimized']<=means['A-reference']*1.005)
    source=Path('/mnt/local/_outputs/deep-llms_th2/q359-proxy-10k-seed1042-20261009-a01/supervised/run/seed-1042')
    matched=Path('/mnt/local/_outputs/deep-llms_th2/q359-A-time-match10800-seed1042-20261010-a01/supervised/run')
    common=None
    for folder in (source/'P6-iso',source/'A',matched/'seed-1042/A'):
        c=read(folder/'train_config.json')
        if (c['world_size']!=8 or c['tokens_per_update']!=1048576 or c['data']['block_size']!=2048
                or c['data']['eval_rows']!=4882 or c['pilot']['attention_backend']!='fa4'
                or any(c['training'][k]!=v for k,v in dict(seed=1042,data_seed=1042,max_steps=28600,
                    warmup_steps=1430,per_device_train_batch_size=16,gradient_accumulation_steps=4).items())):
            raise ValueError('Unexpected source training recipe')
        normalized=comparable_config(c);normalized['training']['save_total_limit']=0
        if common is not None and common!=normalized:raise ValueError('Source recipes differ')
        common=normalized
    prior=read(matched/'time-comparison.json')
    if prior['status']!='passed':raise ValueError('Missing verified original cumulative runtimes')
    p6_prior=read(source/'P6-iso/result.json')['training_cost']['runtime_seconds']
    a_prior=read(source/'A/result.json')['training_cost']['runtime_seconds']
    last=prior['results'][-1]
    if (last['arm']!='A' or last['step']!=10800
            or prior['results'][0]['runtime_seconds']!=a_prior
            or prior['results'][1]['runtime_seconds']!=p6_prior):raise ValueError('Prior runtime identity differs')
    plan=dict(status='prepared',optimized=optimized,benchmark=str(benchmark),
              p6_prior=p6_prior,a_token_prior=a_prior,a_time_prior=last['runtime_seconds'],
              time_source=str(matched/'seed-1042/A/checkpoint-10800'),
              selection='P6 improves at least0.2%, A regresses at most0.5%; otherwise retain reference for all branches',
              time_basis='Actual cumulative Trainer runtime, including original unoptimized history. No retroactive repricing.')
    write(root/'plan.json',plan)
    branch(root,'P6',source/'P6-iso/checkpoint-10000',1000,optimized)
    branch(root,'A-token',source/'A/checkpoint-10000',1000,optimized)
    branch(root,'smoke',source/'P6-iso/checkpoint-10000',2,optimized)


def validate_smoke(root):
    import torch
    root=Path(root);folder=root/'smoke';result=read(folder/'result.json')
    state=read(folder/'checkpoint-10002/trainer_state.json')
    scheduler=torch.load(folder/'checkpoint-10002/scheduler.pt',map_location='cpu',weights_only=True)
    if (result['global_step']!=10002 or not result['cooldown']['complete'] or state['max_steps']!=28600
            or scheduler['last_epoch']!=10002 or any(lr!=0 for lr in scheduler['_last_lr'])):
        raise ValueError('Real eight-GPU cooldown smoke failed')
    if not math.isfinite(result['evaluation']['eval_lm_loss']):raise ValueError('Nonfinite smoke loss')
    write(root/'smoke-verified.json',dict(status='passed',step=10002,final_lr=scheduler['_last_lr']))


def prepare_time(root):
    root=Path(root);plan=read(root/'plan.json')
    p6=read(root/'P6/result.json');a=read(root/'A-token/result.json')
    for r in (p6,a):
        if not r['cooldown']['complete'] or r['global_step']!=11000:raise ValueError('Incomplete initial cooldown')
    p6_total=plan['p6_prior']+p6['training_cost']['runtime_seconds']
    a_rate=a['training_cost']['runtime_seconds']/1000
    steps=duration_for_budget(p6_total,plan['a_time_prior'],a_rate)
    branch(root,'A-time',plan['time_source'],steps,plan['optimized'])
    write(root/'time-budget.json',dict(status='prepared',steps=steps,p6_cumulative_seconds=p6_total,
          a_prior_seconds=plan['a_time_prior'],a_observed_seconds_per_step=a_rate,
          planned_stop=10800+steps,selection='Runtime only; no metric-based step selection'))


def summary(root):
    import torch
    from deep_kv.report import comparable_config
    root=Path(root);plan=read(root/'plan.json');rows={};identity=None
    for name,prior in [('P6',plan['p6_prior']),('A-token',plan['a_token_prior']),('A-time',plan['a_time_prior'])]:
        folder=root/name;config=read(folder/'train_config.json');r=read(folder/'result.json')
        end=r['cooldown']['end'];checkpoint=folder/f'checkpoint-{end}'
        state=read(checkpoint/'trainer_state.json')
        schedule=torch.load(checkpoint/'scheduler.pt',map_location='cpu',weights_only=True)
        if (not r['cooldown']['complete'] or r['global_step']!=end or state['global_step']!=end
                or schedule['last_epoch']!=end or any(lr!=0 for lr in schedule['_last_lr'])
                or r['training_cost']['optimizer_steps']!=r['cooldown']['steps']
                or r['evaluation']['eval_rows']!=4882 or config['training']['save_total_limit']!=0):
            raise ValueError('Incomplete cooldown/evaluation or nonzero final LR')
        for filename in ['model.safetensors','optimizer.pt',*[f'rng_state_{rank}.pth' for rank in range(8)]]:
            if not (checkpoint/filename).is_file() or not (checkpoint/filename).stat().st_size:
                raise ValueError('Incomplete final checkpoint state')
        common=comparable_config(config)
        for key in ('cooldown_start','cooldown_steps'):common['pilot'].pop(key)
        if identity is not None and identity!=common:raise ValueError('Unmatched cooldown data/recipe')
        identity=common
        receipt=read(folder/'cooldown_source.json')
        for relative,expected in receipt['files'].items():
            if digest(Path(receipt['source_checkpoint'])/relative)!=expected:
                raise ValueError('Original checkpoint changed')
        nll=r['evaluation']['eval_lm_loss']
        if not math.isfinite(nll):raise ValueError('Nonfinite evaluation')
        rows[name]=dict(step=end,nll=nll,ppl=math.exp(nll),cooldown=r['cooldown'],
                        cumulative_runtime_seconds=prior+r['training_cost']['runtime_seconds'])
    gap=rows['A-time']['cumulative_runtime_seconds']/rows['P6']['cumulative_runtime_seconds']-1
    write(root/'summary.json',dict(status='passed',results=rows,time_gap_fraction=gap,within_one_percent=abs(gap)<=.01,
        original_checkpoints_unchanged=True,plan=plan,
        note='Single-seed controlled schedule branches. Time matching is approximate and reported explicitly; not a new architecture screen.'))


def make_jobs(root, benchmark):
    """Nested sequential runner, supervised by the same train→burn owner."""
    root=Path(root);project=Path.cwd();study='{run_dir}/branches'
    items=[dict(name='prepare',argv=['{python}','-m','scripts.p6_cooldown_study','prepare',
        '--root',study,'--benchmark',str(benchmark)],required_outputs=[dict(path='branches/plan.json',json_equals=dict(status='prepared'))])]
    for name in ('smoke','P6','A-token','A-time'):
        if name=='A-time':
            items.append(dict(name='time-budget',argv=['{python}','-m','scripts.p6_cooldown_study','prepare-time','--root',study],
                required_outputs=[dict(path='branches/time-budget.json',json_equals=dict(status='prepared'))]))
        items.append(dict(name='cooldown-'+name,gpus=list(range(8)),timeout_seconds=10800,
            argv=['{python}','-m','accelerate.commands.launch','--config_file',str(project/'resources/accelerate_config.yaml'),
                  str(project/'train.py'),study+'/'+name+'.json'],
            required_outputs=[dict(path='branches/'+name+'/result.json')]))
        if name=='smoke':
            items.append(dict(name='validate-smoke',argv=['{python}','-m','scripts.p6_cooldown_study','validate-smoke','--root',study],
                required_outputs=[dict(path='branches/smoke-verified.json',json_equals=dict(status='passed'))]))
    items.append(dict(name='validate',argv=['{python}','-m','scripts.p6_cooldown_study','summary','--root',study],
        required_outputs=[dict(path='branches/summary.json',json_equals=dict(status='passed',original_checkpoints_unchanged=True))]))
    write(root/'cooldown-jobs.json',dict(jobs=items))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','prepare-time','summary','make-jobs','validate-smoke'])
    p.add_argument('--root',required=True);p.add_argument('--benchmark')
    args=p.parse_args()
    if args.command in ('prepare','make-jobs'):
        if not args.benchmark:p.error('--benchmark is required')
        globals()[args.command.replace('-','_')](args.root,args.benchmark)
    else:globals()[args.command.replace('-','_')](args.root)
