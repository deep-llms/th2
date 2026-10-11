"""Fresh matched-seed A/P6 fits followed by preserved 1,000-update cooldowns."""
import argparse
import json
import math
from pathlib import Path
import time

from scripts.proxy_10k_fresh_queue import make as fresh_jobs, ARMS
from scripts.p6_cooldown_study import branch
from scripts.stage_deep_kv_resume import digest
from run_experiments import load_jobs


def read(path):
    return json.loads(Path(path).read_text())


def predecessor_ready(root, guard=Path('/mnt/local/_gpu_guard/DISABLED')):
    """Read-only success gate; GPU ownership is reverified by the launcher later."""
    root=Path(root)/'supervised'
    path=root/'supervisor.json'
    if not path.is_file():
        return False
    state=read(path)
    if state.get('training_status')=='failed':
        raise RuntimeError('Predecessor failed; replication will not launch')
    if not state.get('finished_at'):
        return False
    if (state.get('training_status')!='ok' or state.get('training_returncode')!=0
            or not state.get('burn',{}).get('collective_progress_verified')):
        raise RuntimeError('Predecessor completion/burn validation failed')
    run=read(root/'run/run.json')
    summary=read(root/'run/branches/summary.json')
    burn=read(root/'burn-verified.json')
    if (run['status']!='ok' or summary['status']!='passed'
            or not summary['original_checkpoints_unchanged']
            or not burn.get('collective_progress_verified')
            or burn.get('session')!=state['burn'].get('session')):
        raise RuntimeError('Predecessor artifacts do not establish successful handoff')
    return not Path(guard).exists()


def wait_for_predecessor(root):
    for attempt in range(1440):  # At most twelve hours; never touches GPUs.
        if predecessor_ready(root):
            print('PREDECESSOR_SUCCESS_AND_BURN_VERIFIED',str(root),flush=True)
            return
        if attempt%10==0:
            print('WAITING_FOR_COOLDOWN_SUCCESS_AND_BURN',attempt,flush=True)
        time.sleep(30)
    raise RuntimeError('Timed out waiting for predecessor; no GPU action taken')


def make(recipe, output, seed):
    output=Path(output)
    items=fresh_jobs(recipe,output.with_name('fresh-jobs.json'),seed)
    project=Path(__file__).resolve().parents[1]
    items.append(dict(name='prepare-cooldowns',argv=['{python}','-m','scripts.proxy_replication_queue',
        'prepare','--root','{run_dir}','--seed',str(seed)],
        required_outputs=[dict(path='cooldown/prepared.json',json_equals=dict(status='prepared'))]))
    for arm in ARMS:
        folder=f'cooldown/seed-{seed}/{arm}'
        items.append(dict(name='cooldown-'+arm,gpus=list(range(8)),timeout_seconds=10800,
            argv=['env',f'PYTHONHASHSEED={seed}','{python}','-m','accelerate.commands.launch',
                  '--config_file',str(project/'resources/accelerate_config.yaml'),str(project/'train.py'),
                  '{run_dir}/'+folder+'.json'],required_outputs=[dict(path=folder+'/result.json',
                      json_equals=dict(arm=arm,global_step=11000,status='stopped'))]))
    items.append(dict(name='validate-cooldowns',argv=['{python}','-m','scripts.proxy_replication_queue',
        'validate','--root','{run_dir}','--seed',str(seed)],
        required_outputs=[dict(path='cooldown/summary.json',json_equals=dict(status='passed',original_checkpoints_unchanged=True))]))
    with output.open('x') as handle:
        json.dump(dict(jobs=items),handle,indent=2)
    load_jobs(output)
    return items


def prepare(root,seed):
    root=Path(root)
    for arm in ARMS:
        source=root/f'seed-{seed}'/arm
        config=read(source/'train_config.json');result=read(source/'result.json')
        if (result['global_step']!=10000 or result['arm']!=arm
                or config['training']['seed']!=seed or config['training']['data_seed']!=seed
                or config['training']['save_total_limit']!=0
                or not config['pilot'].get('proxy_sync_optimization')):
            raise ValueError('Unexpected replication source recipe')
        if arm!='A' and config['pilot']['proxy_module_seed']!=seed+1:
            raise ValueError('Unexpected proxy initialization seed')
        branch(root/f'cooldown/seed-{seed}',arm,source/'checkpoint-10000',1000,True)
    (root/'cooldown/prepared.json').write_text(json.dumps(dict(status='prepared',seed=seed)))


def validate(root,seed):
    import torch
    from types import SimpleNamespace
    from scripts.check_fa4_proxy import validate as validate_model
    from deep_kv.report import report
    root=Path(root);rows={}
    validate_model(SimpleNamespace(run_dir=str(root/'cooldown'),seed=seed,steps=11000,
        arms=ARMS,attention_backend='fa4',output=str(root/'cooldown/backend-verified.json')))
    report(root/f'cooldown/seed-{seed}',ARMS,expected_step=11000,expected_seed=seed,expected_attention_backend='fa4')
    for arm in ARMS:
        folder=root/f'cooldown/seed-{seed}'/arm;r=read(folder/'result.json')
        config=read(folder/'train_config.json');checkpoint=folder/'checkpoint-11000'
        scheduler=torch.load(checkpoint/'scheduler.pt',map_location='cpu',weights_only=True)
        if (r['cooldown']!=dict(start=10000,steps=1000,end=11000,complete=True)
                or scheduler['last_epoch']!=11000 or any(lr!=0 for lr in scheduler['_last_lr'])
                or config['training']['save_total_limit']!=0 or r['evaluation']['eval_rows']!=4882
                or r['training_cost']['optimizer_steps']!=1000):
            raise ValueError('Incomplete cooldown, wrong evaluation, or final LR is not zero')
        for filename in ['model.safetensors','optimizer.pt',*[f'rng_state_{rank}.pth' for rank in range(8)]]:
            if not (checkpoint/filename).is_file() or not (checkpoint/filename).stat().st_size:
                raise ValueError('Incomplete final checkpoint')
        receipt=read(folder/'cooldown_source.json')
        for relative,expected in receipt['files'].items():
            if digest(Path(receipt['source_checkpoint'])/relative)!=expected:
                raise ValueError('Original 10k checkpoint changed')
        prior=read(root/f'seed-{seed}'/arm/'result.json')
        nll=r['evaluation']['eval_lm_loss']
        if not math.isfinite(nll):raise ValueError('Nonfinite validation loss')
        rows[arm]=dict(step=11000,nll=nll,ppl=math.exp(nll),pre_cooldown_nll=prior['evaluation']['eval_lm_loss'],
            cumulative_runtime_seconds=prior['training_cost']['runtime_seconds']+r['training_cost']['runtime_seconds'])
    (root/'cooldown/summary.json').write_text(json.dumps(dict(status='passed',seed=seed,results=rows,
        original_checkpoints_unchanged=True,note='Paired seed replication at equal steps/tokens; not an equal-time comparison.'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','validate'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();globals()[a.command](a.root,a.seed)
