#1 +60+a
#th2-tjx3-proxy-fa4-smoke-monitor-20261005-a04
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
/mnt/local/conda-py311/envs/attention_bench/bin/python -u - <<'PYMON'
import json,time,hashlib,re
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect,process
from scripts.train_then_burn import approved_launcher,BURN_HASH,burn_progress
root=Path('/mnt/local/_outputs/deep-llms_th2/proxy-fa4-smoke-20261005-a01')
def tail(path,n=3500):
    if not path.is_file():return 'MISSING'
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-n));return stream.read().decode(errors='replace')
print('GPU_STATUS',json.dumps(inspect()),flush=True)
from accelerate.commands.config.config_args import default_yaml_config_file,load_config_from_file
import importlib.metadata as md
source=Path('resources/accelerate_config.yaml');destination=Path(default_yaml_config_file)
print('ACCELERATE_VERIFIED',json.dumps(dict(path=str(destination),matches_resource=source.read_bytes()==destination.read_bytes(),
    sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),config=load_config_from_file(str(destination)).to_dict(),
    versions={name:md.version(name) for name in ('torch','transformers','accelerate','datasets','flash-attn-4','nvidia-cutlass-dsl')})),flush=True)
for name in ['inspection.json','supervised/reclaim.json','supervised/gpus-free-before-training.json',
             'supervised/supervisor.json','supervised/run/run.json','supervised/run/numerics/summary.json',
             'supervised/run/smoke-validation.json','supervised/run/complete.json','supervised/burn-verified.json']:
    p=root/name
    if p.is_file():
        raw=p.read_bytes();print('ARTIFACT',json.dumps(dict(path=name,sha256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))),flush=True)
    else:print('MISSING_ARTIFACT',name,flush=True)
print('LAUNCH_TAIL',tail(root/'launch.log',6000),flush=True)
run=root/'supervised/run'
for path in sorted(run.glob('*.log')):
    print('JOB_TAIL',path.name,tail(path,2400),flush=True)
for path in sorted((run/'numerics').glob('case-*.log')):
    print('NUMERICS_TAIL',path.name,tail(path,2400),flush=True)
for path in sorted((run/'seed-42').glob('*/result.json')):
    value=json.loads(path.read_text());print('ARM_RESULT',json.dumps(dict(arm=value['arm'],step=value['global_step'],
        status=value['status'],cost=value.get('training_cost'),evaluation=value['evaluation'])),flush=True)
burn=root/'supervised/burn.log'
if burn.is_file():
    before=burn.read_text();time.sleep(12);after=burn.read_text();status=inspect()
    approved={str(Path.cwd()/'resources/llm_pretrain_burn.py'):BURN_HASH}
    known=len(status['workers'])==8 and all(approved_launcher(process(pid)['ppid'],approved) for pid in status['workers'])
    progress=lambda value:re.findall(r'gpu_burn_progress rank=0 completed_cycles=(\d+).*?completed_collective_payload_gib=([\d.]+)',value)
    a,b=progress(before),progress(after)
    advancing=bool(a and b and int(b[-1][0])>int(a[-1][0]) and float(b[-1][1])>float(a[-1][1]))
    print('FINAL_BURN',json.dumps(dict(at=status['time'],known_eight_workers=known,ready=burn_progress(after),advancing=advancing,guard_released=not status['guard_disabled'],gpus=status['gpus'])),flush=True)
PYMON
