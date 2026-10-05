#1 +60+a
#th2-tjx3-p1-alpha1-status-20261006-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
/mnt/local/conda-py311/envs/attention_bench/bin/python -u - <<'PY'
import json,hashlib
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect,process
from scripts.train_then_burn import burn_progress,approved_launcher,APPROVED_BURNS,BURN_HASH
from deep_kv.report import comparable_config
root=Path('/mnt/local/_outputs/deep-llms_th2/proxy-p1-alpha1-2500-20261005-a01')
run=root/'supervised/run';arm=run/'training/seed-42/P1-block'
def artifact(path):
    if path.is_file():
        raw=path.read_bytes()
        print('ARTIFACT',json.dumps(dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))),flush=True)
def tail(path,n=5000):
    if not path.is_file():return 'MISSING'
    with path.open('rb') as f:f.seek(max(0,path.stat().st_size-n));return f.read().decode(errors='replace')
status=inspect();print('GPU_STATUS',json.dumps(status),flush=True)
for name in ('supervised/supervisor.json','supervised/run/run.json','supervised/run/complete.json','supervised/burn-verified.json',
             'supervised/run/smoke/validation.json','supervised/run/training/validation.json'):
    artifact(root/name)
for name in ('result.json','eval_results.json','train_config.json'):artifact(arm/name)
if (arm/'train_config.json').is_file():
    prior=Path('/mnt/local/_outputs/deep-llms_th2/proxy-fa4-screen-seed42-2500-20261005-a02/supervised/run/seed-42/P1-block/train_config.json')
    value=json.loads((arm/'train_config.json').read_text());alpha=value['pilot'].pop('proxy_alpha_init')
    print('MATCHED_PRIOR_P1_EXCEPT_ALPHA',alpha,comparable_config(value)==comparable_config(json.loads(prior.read_text())),flush=True)
for path in sorted(arm.glob('checkpoint-*')):
    names=['model.safetensors','optimizer.pt','scheduler.pt','trainer_state.json',*[f'rng_state_{i}.pth' for i in range(8)]]
    print('CHECKPOINT',path.name,json.dumps({n:(path/n).stat().st_size if (path/n).is_file() else None for n in names}),flush=True)
if (arm/'trainer_state.json').is_file():
    state=json.loads((arm/'trainer_state.json').read_text())
    print('TRAINER_STATE',json.dumps(dict(global_step=state['global_step'],max_steps=state['max_steps'],last_logs=state['log_history'][-3:])),flush=True)
print('TRAINING_LOG',tail(run/'training-seed-42-arm-P1-block.log',12000),flush=True)
print('LAUNCH_LOG',tail(root/'launch.log',3000),flush=True)
burn=root/'supervised/burn.log'
print('BURN_LOG',tail(burn,4000),flush=True)
if burn.is_file():
    approved={**APPROVED_BURNS,str(Path.cwd()/'resources/llm_pretrain_burn.py'):BURN_HASH}
    identities=bool(status['workers']) and all(approved_launcher(pid,approved) or approved_launcher(process(pid)['ppid'],approved) for pid in status['workers'])
    print('BURN_CHECK',json.dumps(dict(verified_burn_workers=identities,count=len(status['workers']),guard_disabled=status['guard_disabled'],collective_progress=burn_progress(burn.read_text()))),flush=True)
PY
