#1 +120+a
#th2-tjx3-proxy-baseline-fa4-startup-monitor-20261005-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
export CUDA_VISIBLE_DEVICES=''
/mnt/local/conda-py311/envs/train_env/bin/python -u - <<'PY'
import json,time
from pathlib import Path
from run_experiments import now
from scripts.verified_gpu_reclaim import inspect
root=Path('/mnt/local/_outputs/deep-llms_th2/proxy-baseline-A-fa4-2500-20261005-a01')
run=root/'supervised/run'
def tail(path,limit=3500):
    if not path.is_file():return ''
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-limit));return f.read().decode(errors='replace')
for name in ['preflight.json','supervised/reclaim.json','supervised/gpus-free-before-training.json']:
    path=root/name
    if path.is_file():print('STARTUP_RECEIPT',name,path.read_text(),flush=True)
for tick in range(60):
    print('READ_ONLY_STATUS',now(),flush=True)
    print('LAUNCH_TAIL',tail(root/'launch.log',1200),flush=True)
    if (run/'run.json').is_file():print('QUEUE_STATE',(run/'run.json').read_text(),flush=True)
    for name in ['production-numerics.log','smoke-seed-42-arm-A.log','validate-smoke.log','baseline-seed-42-arm-A.log']:
        value=tail(run/name)
        if value:print('JOB_LOG',name,value,flush=True)
    numerics=run/'numerics.json'
    if numerics.is_file():print('NUMERICAL_VALIDATION',numerics.read_text(),flush=True)
    smoke=run/'smoke/validation.json'
    if smoke.is_file():print('SMOKE_VALIDATION',smoke.read_text(),flush=True)
    receipt=root/'supervised/supervisor.json'
    if receipt.is_file():
        value=json.loads(receipt.read_text())
        if value.get('finished_at'):
            print('SUPERVISOR_FINISHED',json.dumps(value),flush=True);break
    baseline=tail(run/'baseline-seed-42-arm-A.log',20000)
    if smoke.is_file() and 'step_lm_loss' in baseline:
        print('BASELINE_OPTIMIZER_UPDATES_OBSERVED',flush=True);break
    if tick%4==0:print('GPU_STATUS',json.dumps(inspect()),flush=True)
    time.sleep(30)
else:print('MONITOR_WINDOW_ENDED_NO_PROCESS_CHANGES',flush=True)
print('FINAL_GPU_STATUS',json.dumps(inspect()),flush=True)
PY
