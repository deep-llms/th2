#!/usr/bin/env bash
# Wait for accepted timings, then run preserved-checkpoint cooldown branches.
set -euo pipefail
TASK_ROOT=$1
TASK_HOST=${2:?Pass verified hostname}
TASK_BENCHMARK=${3:?Pass exact synchronization trial root}
cd /mnt/local/deep-llms_th2
test "$(hostname)" = "$TASK_HOST"
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate attention_bench
export NCCL_NVLS_ENABLE=0 WANDB_MODE=offline WANDB_PROJECT=deep2shallow
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=1042 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -u - "$TASK_ROOT" "$TASK_HOST" "$TASK_BENCHMARK" <<'PY'
import copy,importlib.metadata as md,json,os,shutil,subprocess,sys
from pathlib import Path
from accelerate.commands.config.config_args import default_yaml_config_file,load_config_from_file
from scripts.verified_gpu_reclaim import inspect,process
from scripts.train_then_burn import approved_launcher,APPROVED_BURNS,BURN_HASH,GUARD_HASH,digest
from run_experiments import load_jobs
from deep_kv.__main__ import jobs

root,host=Path(sys.argv[1]),sys.argv[2];project=Path.cwd()
benchmark=Path(sys.argv[3]);supervised=benchmark/'supervised'
# Read-only waiting never touches the current allocation or its guard.
import time
for attempt in range(240):
    path=supervised/'supervisor.json'
    state=json.loads(path.read_text()) if path.is_file() else {}
    if state.get('training_status')=='failed':raise RuntimeError('Synchronization study failed; cooldown not launched')
    if state.get('finished_at'):
        assert state['training_status']=='ok' and state['burn']['collective_progress_verified']
        assert json.loads((supervised/'run/summary.json').read_text())['status']=='passed'
        if not Path('/mnt/local/_gpu_guard/DISABLED').exists():break
    if attempt%10==0:print('WAITING_FOR_SYNC_ACCEPTANCE',attempt,flush=True)
    time.sleep(30)
else:raise RuntimeError('Timed out waiting for accepted benchmark; no GPU action taken')
for p,v in [('torch','2.14.1'),('transformers','5.9.0'),('accelerate','1.13.0'),('datasets','4.8.5'),
            ('flash-attn-4','4.0.0b33'),('nvidia-cutlass-dsl','4.8.0'),('wandb','0.30.0')]:
    assert md.version(p).split('+')[0]==v,(p,md.version(p))
assert shutil.disk_usage(root).free>200*2**30
with (root/'cpu-tests.log').open('x') as log:
    subprocess.run([sys.executable,'-m','unittest','tests.test_proxy_sync','tests.test_cooldown','-v'],
        stdout=log,stderr=subprocess.STDOUT,check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
checkpoint=Path('/mnt/local/_outputs/deep-llms_th2/q359-proxy-10k-seed1042-20261009-a01/supervised/run/seed-1042/P6-iso/checkpoint-10000')
expected='3e04cc0a27f1e0d5a1e97efdc1e34c11d4e821f9b34fb76898a0652fd8d28442'
assert digest(checkpoint/'model.safetensors')==expected
assert json.loads((checkpoint/'trainer_state.json').read_text())['global_step']==10000
source=project/'resources/accelerate_config.yaml';destination=Path(default_yaml_config_file)
destination.parent.mkdir(parents=True,exist_ok=True)
if destination.exists():shutil.copy2(destination,root/'accelerate.previous.yaml')
shutil.copy2(source,destination);assert source.read_bytes()==destination.read_bytes()
c=load_config_from_file(str(destination)).to_dict()
assert c['num_processes']==8 and c['mixed_precision']=='bf16' and c['distributed_type']=='MULTI_GPU'
subprocess.run(['accelerate','env'],check=True)
from scripts.p6_cooldown_study import make_jobs
make_jobs(root,supervised/'run')
items=load_jobs(root/'cooldown-jobs.json')
inspection=inspect();assert inspection['host']==host and not inspection['guard_disabled']
assert len(inspection['gpus'])==8 and all('B200' in g['name'] for g in inspection['gpus'])
assert digest('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
approved={**APPROVED_BURNS,str(project/'resources/llm_pretrain_burn.py'):BURN_HASH}
assert all(approved_launcher(pid,approved) or approved_launcher(process(pid)['ppid'],approved) for pid in inspection['workers'])
(root/'inspection.json').write_text(json.dumps(inspection,indent=2))
(root/'preflight.json').write_text(json.dumps(dict(status='passed',checkpoint=str(checkpoint),
    checkpoint_sha256=expected,accelerate_sha256=digest(destination),stages=len(items)),indent=2))
print('P6_COOLDOWN_PREFLIGHT_PASSED',flush=True)
PY
exec python -u -m scripts.train_then_burn --config "$TASK_ROOT/cooldown-jobs.json" --output "$TASK_ROOT/supervised" \
  --inspection "$TASK_ROOT/inspection.json" --host "$TASK_HOST" --burn-session "$(basename "$TASK_ROOT")-final-burn"
