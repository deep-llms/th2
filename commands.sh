#1 +60+a
#th2-tjx3-proxy-fa4-screen-monitor-20261005-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
/mnt/local/conda-py311/envs/attention_bench/bin/python -u - <<'PY'
import json,hashlib,re,time
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect
from deep_kv.report import comparable_config
from accelerate.commands.config.config_args import default_yaml_config_file,load_config_from_file
root=Path('/mnt/local/_outputs/deep-llms_th2/proxy-fa4-screen-seed42-2500-20261005-a01')
base=Path('/mnt/local/_outputs/deep-llms_th2/proxy-baseline-A-fa4-2500-20261005-a01/supervised/run/baseline/seed-42/A')
def read(path):return json.loads(path.read_text())
def tail(path,n=8000):
    if not path.is_file():return 'MISSING'
    with path.open('rb') as f:f.seek(max(0,path.stat().st_size-n));return f.read().decode(errors='replace')
time.sleep(30)
print('GPU_STATUS',json.dumps(inspect()),flush=True)
config=Path(default_yaml_config_file)
print('ACCELERATE',json.dumps(dict(path=str(config),matches_resource=config.read_bytes()==Path('resources/accelerate_config.yaml').read_bytes(),config=load_config_from_file(str(config)).to_dict())),flush=True)
for name in ['inspection.json','revalidated-smokes.json','supervised/reclaim.json','supervised/gpus-free-before-training.json',
             'supervised/supervisor.json','supervised/run/run.json','supervised/run/complete.json','supervised/burn-verified.json']:
    p=root/name
    if p.is_file():
        raw=p.read_bytes();print('ARTIFACT',json.dumps(dict(path=name,sha256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))),flush=True)
print('LAUNCH',tail(root/'launch.log',14000),flush=True)
run=root/'supervised/run'
for p in sorted(run.glob('*.log')):print('JOB',p.name,tail(p,9000),flush=True)
for p in sorted((run/'seed-42').glob('*/train_config.json')):
    value=read(p)
    print('MATCHES_BASELINE',p.parent.name,comparable_config(value)==comparable_config(read(base/'train_config.json')),flush=True)
    for receipt in sorted(p.parent.glob('fa4-*.json')):print('BACKEND_RECEIPT',json.dumps(read(receipt)),flush=True)
print('BURN_TAIL',tail(root/'supervised/burn.log',3000),flush=True)
PY
