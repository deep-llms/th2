#1 +60+a
#th2-tjx3-proxy-gate-sweep-monitor-20261005-a02
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
sleep 40
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
/mnt/local/conda-py311/envs/attention_bench/bin/python -u - <<'PY'
import json,hashlib
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect
from scripts.train_then_burn import burn_progress
root=Path('/mnt/local/_outputs/deep-llms_th2/proxy-gate-sweep-20261005-a01')
def artifact(p):
    if p.is_file():
        raw=p.read_bytes();print('ARTIFACT',json.dumps(dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))),flush=True)
def tail(p,n=5000):
    if not p.is_file():return 'MISSING'
    with p.open('rb') as f:f.seek(max(0,p.stat().st_size-n));return f.read().decode(errors='replace')
print('GPU_STATUS',json.dumps(inspect()),flush=True)
for name in ('inspection.json','supervised/reclaim.json','supervised/gpus-free-before-training.json',
             'supervised/supervisor.json','supervised/run/run.json','supervised/run/complete.json',
             'supervised/burn-verified.json'):
    artifact(root/name)
for arm in ('P1-block','P3-block'):
    folder=root/'supervised/run'/arm
    artifact(folder/'summary.json')
    for p in folder.glob('fa4-*.json'):artifact(p)
    print('JOB',arm,tail(root/'supervised/run'/f'evaluate-{arm}.log',10000),flush=True)
print('LAUNCH',tail(root/'launch.log',6000),flush=True)
print('BURN',tail(root/'supervised/burn.log',4000),flush=True)
p=root/'supervised/burn.log'
if p.is_file():print('BURN_PROGRESS_VERIFIED',burn_progress(p.read_text()),flush=True)
PY
