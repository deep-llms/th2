#1 +60+a
#th2-tjx3-fa4-A-completion-status-20261005-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
/mnt/local/conda-py311/envs/train_env/bin/python -u - <<'PY'
import json,time,re
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect,process
from scripts.train_then_burn import approved_launcher,BURN_HASH,burn_progress
root=Path('/mnt/local/_outputs/deep-llms_th2/proxy-baseline-A-fa4-2500-20261005-a01')
run=root/'supervised/run';arm=run/'baseline/seed-42/A'
def tail(path,n=2500):
    if not path.is_file():return ''
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-n));return stream.read().decode(errors='replace')
print('GPU_STATUS_BEFORE',json.dumps(inspect()),flush=True)
for name in ('supervised/supervisor.json','supervised/burn-verified.json','supervised/run/run.json',
             'supervised/run/baseline/validation.json','supervised/run/baseline/seed-42/A/result.json'):
    path=root/name
    print('ARTIFACT',name,path.read_text() if path.is_file() else 'MISSING',flush=True)
state=arm/'trainer_state.json'
if state.is_file():
    state=json.loads(state.read_text());rows=[r for r in state['log_history'] if 'loss' in r]
    print('TRAINING_STATE',json.dumps(dict(global_step=state['global_step'],max_steps=state['max_steps'],last_logs=rows[-5:])),flush=True)
checkpoint=arm/'checkpoint-2500'
files=['model.safetensors','optimizer.pt','scheduler.pt','trainer_state.json',*[f'rng_state_{i}.pth' for i in range(8)]]
print('CHECKPOINT_FILES',json.dumps({name:(checkpoint/name).stat().st_size if (checkpoint/name).is_file() else None for name in files}),flush=True)
for name in ('launch.log','supervised/run/baseline-seed-42-arm-A.log'):
    print('LOG_TAIL',name,tail(root/name),flush=True)
old=Path('/mnt/local/_outputs/deep-llms_th2/proxy-baseline-A-2500-20261005-a01/supervised/run/baseline/seed-42/A/result.json')
if old.is_file() and (arm/'result.json').is_file():
    a=json.loads(old.read_text());b=json.loads((arm/'result.json').read_text())
    print('MATCHED_LOSS_COMPARISON',json.dumps(dict(dense=a['evaluation']['eval_lm_loss'],fa4=b['evaluation']['eval_lm_loss'],
          difference_fa4_minus_dense=b['evaluation']['eval_lm_loss']-a['evaluation']['eval_lm_loss'])),flush=True)
burn=root/'supervised/burn.log'
before=burn.read_text() if burn.is_file() else ''
time.sleep(12)
after=burn.read_text() if burn.is_file() else ''
status=inspect();print('GPU_STATUS_AFTER',json.dumps(status),flush=True)
approved={str(Path.cwd()/'resources/llm_pretrain_burn.py'):BURN_HASH}
workers=status['workers']
known=len(workers)==8 and all(approved_launcher(process(pid)['ppid'],approved) for pid in workers)
progress=lambda value:re.findall(r'gpu_burn_progress rank=0 completed_cycles=(\d+).*?completed_collective_payload_gib=([\d.]+)',value)
x,y=progress(before),progress(after)
advancing=bool(x and y and int(y[-1][0])>int(x[-1][0]) and float(y[-1][1])>float(x[-1][1]))
print('CURRENT_BURN_VERIFICATION',json.dumps(dict(known_eight_workers=known,all_rank_readiness=burn_progress(after),
      newly_advancing_collectives=advancing,before=x[-1:] or None,after=y[-1:] or None,guard_released=not status['guard_disabled'])),flush=True)
print('BURN_TAIL',tail(burn),flush=True)
PY
