#1 +60+a
#th2-tjx3-proxy-baseline-status-20261005-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
export CUDA_VISIBLE_DEVICES=''
/mnt/local/conda-py311/envs/train_env/bin/python -u - <<'PY'
import json,time
from pathlib import Path
from run_experiments import now
from scripts.verified_gpu_reclaim import inspect
root=Path('/mnt/local/_outputs/deep-llms_th2/proxy-baseline-A-2500-20261005-a01')
run=root/'supervised/run'
def tail(path,limit=6000):
    if not path.is_file():return 'MISSING'
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-limit));return f.read().decode(errors='replace')
print('READ_ONLY_STATUS',now())
for name in ['supervised/supervisor.json','supervised/burn-verified.json','supervised/run/run.json','supervised/run/complete.json','supervised/run/smoke/validation.json','supervised/run/baseline/seed-42/A/result.json','supervised/run/baseline/seed-42/A/eval_results.json']:
    path=root/name
    print('RECEIPT',name,path.read_text() if path.is_file() else 'MISSING')
for name in ['launch.log','supervised/run/smoke-seed-42-arm-A.log','supervised/run/validate-smoke.log','supervised/run/baseline-seed-42-arm-A.log','supervised/burn.log']:
    print('LOG_TAIL',name,tail(root/name))
model=run/'baseline/seed-42/A'
for path in sorted(model.glob('checkpoint-*')):
    print('CHECKPOINT',path.name,[(p.name,p.stat().st_size) for p in path.iterdir() if p.is_file()])
for path in [model/'trainer_state.json',model/'checkpoint-2500/trainer_state.json']:
    if path.is_file():
        state=json.loads(path.read_text());state['log_history']=state.get('log_history',[])[-5:]
        print('TRAINER_STATE',str(path),json.dumps(state))
print('GPU_STATUS',json.dumps(inspect()))
print('BURN_TAIL_1',tail(root/'supervised/burn.log',1400))
time.sleep(15)
print('BURN_TAIL_2',tail(root/'supervised/burn.log',1400))
print('FINAL_READ_ONLY_TIME',now())
PY
