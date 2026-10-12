#!/usr/bin/env bash
set -euo pipefail
TASK_SOURCE=${1:?Pass stopped expanded study root}
TASK_ROOT=${2:?Pass fresh destination root}
TASK_HOST=${3:?Pass verified hostname}
cd /mnt/local/deep-llms_th2
test "$(hostname)" = "$TASK_HOST"
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate eval_fa4
export NCCL_NVLS_ENABLE=0 WANDB_MODE=offline WANDB_PROJECT=deep2shallow
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m pip check
python -u - "$TASK_SOURCE" "$TASK_ROOT" "$TASK_HOST" <<'PY'
import json,shutil,subprocess,sys,time
from pathlib import Path
from accelerate.commands.config.config_args import default_yaml_config_file,load_config_from_file
from scripts.reduce_expanded_study import build
from scripts.finetune_study import write
from scripts.train_then_burn import digest,GUARD_HASH,APPROVED_BURNS,BURN_HASH,approved_launcher
from scripts.verified_gpu_reclaim import inspect,process
source,root=map(Path,sys.argv[1:3]);host=sys.argv[3];project=Path.cwd()
for _ in range(90):
    if (source/'reduce-seed42-stop.json').is_file():
        pane=subprocess.run(['tmux','display-message','-p','-t',source.name,'#{pane_dead}'],capture_output=True,text=True)
        if pane.returncode==0 and pane.stdout.strip()=='1':break
    time.sleep(10)
else:raise RuntimeError('Prior supervisor has not completed verified stop/handoff')
receipt=build(source,root,project)
print('SINGLE_SEED_PLAN',json.dumps({k:v for k,v in receipt.items() if k!='artifacts'}),flush=True)
config=project/'resources/accelerate_config.yaml';cache=Path(default_yaml_config_file)
cache.parent.mkdir(parents=True,exist_ok=True)
if cache.exists():shutil.copy2(cache,root/'accelerate.previous.yaml')
shutil.copy2(config,cache)
if config.read_bytes()!=cache.read_bytes():raise ValueError('Accelerate copy differs')
a=load_config_from_file(str(cache)).to_dict()
if a['num_processes']!=8 or a['mixed_precision']!='bf16' or a['distributed_type']!='MULTI_GPU':raise ValueError('Wrong Accelerate config')
subprocess.run(['accelerate','env'],check=True)
status=inspect()
if status['host']!=host or status['guard_disabled'] or len(status['gpus'])!=8 or any('B200' not in g['name'] for g in status['gpus']):raise ValueError('Unexpected GPU state')
if digest('/mnt/local/_gpu_guard/gpu_guard.sh')!=GUARD_HASH:raise ValueError('Unknown GPU guard')
approved={**APPROVED_BURNS,str(project/'resources/llm_pretrain_burn.py'):BURN_HASH}
if any(not (approved_launcher(p,approved) or approved_launcher(process(p)['ppid'],approved)) for p in status['workers']):raise ValueError('Unknown GPU work; no signals sent')
write(root/'inspection.json',status)
write(root/'preflight.json',dict(status='passed',accelerate_sha256=digest(cache),accelerate_cache=str(cache),source=str(source)))
print('REDUCED_STUDY_PREFLIGHT_PASSED',flush=True)
PY
exec python -u -m scripts.train_then_burn --config "$TASK_ROOT/jobs.json" --output "$TASK_ROOT/supervised" \
  --inspection "$TASK_ROOT/inspection.json" --host "$TASK_HOST" --burn-session "$(basename "$TASK_ROOT")-final-burn"
