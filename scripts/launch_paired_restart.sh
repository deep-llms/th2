#!/usr/bin/env bash
set -euo pipefail
TASK_SOURCE=${1:?Pass failed paired study root}
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
from scripts.restart_paired_study import prepare
from scripts.finetune_study import write
from scripts.train_then_burn import digest,GUARD_HASH,APPROVED_BURNS,BURN_HASH,approved_launcher
from scripts.verified_gpu_reclaim import inspect,process
source,root=map(Path,sys.argv[1:3]);host=sys.argv[3];project=Path.cwd()
pane=subprocess.run(['tmux','display-message','-p','-t',source.name,'#{pane_dead}'],capture_output=True,text=True)
if pane.returncode!=0 or pane.stdout.strip()!='1':raise RuntimeError('Prior supervisor is not verified stopped')
prepare(source,root)
print('PAIRED_ONLY_RESTART_PREPARED',flush=True)
# Exercise every paired task's real offline loader before reclaiming GPUs.
from eval.paired import protocol
from eval.benchmarks import local_dataset_paths
from lm_eval.tasks import get_task_dict
jobs=json.loads((root/'jobs.json').read_text())['jobs']
args=jobs[0]['argv'];data=args[args.index('--dataset-root')+1]
mapping=local_dataset_paths(data,root/'manifest.json')
for name in protocol.RULES:
    loaded=get_task_dict(protocol.configs(name,mapping),task_manager=protocol.manager())
    for task in loaded.values():
        rows=task.eval_docs
        if not len(rows):raise ValueError('Empty evaluation task: '+name)
        row=rows[0];task.doc_to_text(row);task.doc_to_target(row)
        if task.config.output_type=='multiple_choice':task.doc_to_choice(row)
    print('PAIRED_DATA_LOAD_PASSED',name,flush=True)
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
print('PAIRED_RESTART_PREFLIGHT_PASSED',flush=True)
PY
exec python -u -m scripts.train_then_burn --config "$TASK_ROOT/jobs.json" --output "$TASK_ROOT/supervised" \
  --inspection "$TASK_ROOT/inspection.json" --host "$TASK_HOST" --burn-session "$(basename "$TASK_ROOT")-final-burn"
