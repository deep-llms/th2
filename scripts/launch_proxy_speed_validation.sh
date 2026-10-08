#!/usr/bin/env bash
# Explicitly authorized disposable B200 validation; invoked by commands.sh.
set -euo pipefail
TASK_ROOT=$1
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-tjx3-worker-0
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate attention_bench
export NCCL_NVLS_ENABLE=0 WANDB_MODE=offline WANDB_PROJECT=deep2shallow
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -u - "$TASK_ROOT" <<'PY'
import hashlib,importlib.metadata as md,json,shutil,subprocess,sys
from pathlib import Path
from accelerate.commands.config.config_args import default_yaml_config_file,load_config_from_file
from scripts.verified_gpu_reclaim import inspect,process
from scripts.train_then_burn import approved_launcher,APPROVED_BURNS,BURN_HASH,GUARD_HASH,digest
from scripts.proxy_speed_queue import make
from scripts.proxy_speed_validation import REFERENCE
from train import load_text
root=Path(sys.argv[1]);project=Path.cwd()
versions={p:md.version(p) for p in ('torch','transformers','accelerate','datasets','flash-attn-4','nvidia-cutlass-dsl')}
for p,v in [('torch','2.14.1'),('transformers','5.9.0'),('accelerate','1.13.0'),('datasets','4.8.5'),('flash-attn-4','4.0.0b33'),('nvidia-cutlass-dsl','4.8.0')]:
 assert versions[p].split('+')[0]==v,(p,versions[p])
assert shutil.disk_usage(root).free>500*2**30
reference=json.loads((REFERENCE/'manifest.json').read_text())
for name,expected in reference['files'].items():assert digest(REFERENCE/name)==expected
recipe=json.loads((project/'proxy_heads.b200.json').read_text())
for key in ('config_name','tokenizer_name','data_dir','eval_data_dir'):assert Path(recipe[key]).exists(),key
data={}
for key in ('data_dir','eval_data_dir'):
 ds=load_text(recipe[key]);assert len(ds)>0 and 'text' in ds.column_names
 data[key]=dict(rows=len(ds),fingerprint=ds._fingerprint)
rows=Path('/mnt/local/_outputs/deep-llms_th2/proxy-final-attention-check-20261005-a01/prepared/train.json')
assert rows.is_file() and len(json.loads(rows.read_text()))==16
source=project/'resources/accelerate_config.yaml';destination=Path(default_yaml_config_file)
destination.parent.mkdir(parents=True,exist_ok=True)
if destination.exists():shutil.copy2(destination,root/'accelerate.previous.yaml')
shutil.copy2(source,destination);assert source.read_bytes()==destination.read_bytes()
config=load_config_from_file(str(destination)).to_dict()
assert config['num_processes']==8 and config['mixed_precision']=='bf16' and config['distributed_type']=='MULTI_GPU'
subprocess.run(['accelerate','env'],check=True)
inspection=inspect()
assert inspection['host']=='thiennh-p6-tjx3-worker-0' and not inspection['guard_disabled']
assert len(inspection['gpus'])==8 and all('B200' in g['name'] for g in inspection['gpus'])
assert digest('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
approved={**APPROVED_BURNS,str(project/'resources/llm_pretrain_burn.py'):BURN_HASH}
assert all(approved_launcher(pid,approved) or approved_launcher(process(pid)['ppid'],approved) for pid in inspection['workers'])
items=make(root,project/'proxy_heads.b200.json',rows)
(root/'inspection.json').write_text(json.dumps(inspection,indent=2))
(root/'preflight.json').write_text(json.dumps(dict(status='passed',versions=versions,data=data,
 environment=sys.executable,reference=reference,rows_sha256=digest(rows),stages=len(items),
 accelerate_cache=str(destination),accelerate_sha256=digest(destination),
 source_sha256={name:digest(project/name) for name in ('deep_kv/proxy.py','deep_kv/proxy_memory.py','deep_kv/proxy_training.py','train.py','scripts/proxy_speed_validation.py','scripts/profile_proxy_training.py')}),indent=2))
print('PROXY_SPEED_PREFLIGHT_PASSED',len(items),inspection['workers'],flush=True)
PY
exec python -u -m scripts.train_then_burn --config "$TASK_ROOT/jobs.json" --output "$TASK_ROOT/supervised" \
 --inspection "$TASK_ROOT/inspection.json" --host thiennh-p6-tjx3-worker-0 \
 --burn-session "$(basename "$TASK_ROOT")-final-burn"
