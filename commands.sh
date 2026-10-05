#1 +60+a
#th2-tjx3-proxy-p1-alpha1-monitor-20261005-a02
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate attention_bench
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
/mnt/local/conda-py311/envs/attention_bench/bin/python -u - <<'PY'
import json,hashlib,re
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect
from deep_kv.report import comparable_config
from accelerate.commands.config.config_args import default_yaml_config_file,load_config_from_file
root=Path('/mnt/local/_outputs/deep-llms_th2/proxy-p1-alpha1-2500-20261005-a01')
base=Path('/mnt/local/_outputs/deep-llms_th2/proxy-fa4-screen-seed42-2500-20261005-a02/supervised/run/seed-42/P1-block')
def read(path):return json.loads(path.read_text())
def tail(path,n=8000):
    if not path.is_file():return 'MISSING'
    with path.open('rb') as f:f.seek(max(0,path.stat().st_size-n));return f.read().decode(errors='replace')
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
for p in sorted(run.glob('*/seed-42/*/train_config.json')):
    value=read(p)
    original=read(base/'train_config.json')
    phase=p.relative_to(run).parts[0]
    alpha=value['pilot'].pop('proxy_alpha_init')
    if phase=='smoke':
        original['training']['logging_steps']=1
        original['data'].update(eval_rows=32,monitor_rows=32)
        original['eval_fingerprint']=value['eval_fingerprint']
    print('MATCHES_PRIOR_EXCEPT_ALPHA',phase,alpha,comparable_config(value)==comparable_config(original),flush=True)
    assert alpha==1. and comparable_config(value)==comparable_config(original)
    print('SAVED_CONFIG',phase,json.dumps(read(p)),flush=True)
for phase in ('smoke','training'):
    p=run/phase/'validation.json'
    if p.is_file():print('VALIDATION',phase,p.read_text(),flush=True)


print('BURN_TAIL',tail(root/'supervised/burn.log',3000),flush=True)
PY
