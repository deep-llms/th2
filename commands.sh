#1 +60+a
#th2-tjx3-p3-block-status-20261005-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
/mnt/local/conda-py311/envs/attention_bench/bin/python -u - <<'PY'
import json,hashlib
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect
from deep_kv.report import comparable_config
root=Path('/mnt/local/_outputs/deep-llms_th2/proxy-fa4-screen-seed42-2500-20261005-a02')
run=root/'supervised/run'
base=Path('/mnt/local/_outputs/deep-llms_th2/proxy-baseline-A-fa4-2500-20261005-a01/supervised/run/baseline/seed-42/A')
def artifact(p):
    if p.is_file():
        raw=p.read_bytes();print('ARTIFACT',json.dumps(dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))),flush=True)
def tail(p,n=5000):
    if not p.is_file():return 'MISSING'
    with p.open('rb') as f:f.seek(max(0,p.stat().st_size-n));return f.read().decode(errors='replace')
print('GPU_STATUS',json.dumps(inspect()),flush=True)
for p in [root/'supervised/supervisor.json',run/'run.json',run/'complete.json',root/'supervised/burn-verified.json']:
    artifact(p)
for arm in ('P1-block','P3-block','P1-lambda0','P3-lambda0','P1-flow','V1','V3'):
    folder=run/'seed-42'/arm
    for p in [folder/'result.json',run/'seed-42'/f'{arm}-validation.json']:artifact(p)
    cfg=folder/'train_config.json'
    if cfg.is_file():
        value=json.loads(cfg.read_text());reference=json.loads((base/'train_config.json').read_text())
        print('MATCHES_BASELINE',arm,comparable_config(value)==comparable_config(reference),value['train_fingerprint'],value['eval_fingerprint'],flush=True)
    checkpoint=folder/'checkpoint-2500'
    if checkpoint.is_dir():print('CHECKPOINT',arm,json.dumps({p.name:p.stat().st_size for p in checkpoint.iterdir() if p.is_file()}),flush=True)
    log=run/f'seed-42-arm-{arm}.log'
    if log.is_file():print('JOB',arm,tail(log),flush=True)
print('LAUNCH',tail(root/'launch.log',3000),flush=True)
print('BURN',tail(root/'supervised/burn.log',1500),flush=True)
PY
