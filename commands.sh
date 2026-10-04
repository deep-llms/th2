#1 +60+a
#th2-tjx3-baseline-preflight-20261005-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline NCCL_NVLS_ENABLE=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/mnt/local/conda-py311/envs/train_env/bin/python -u - <<'PY'
import importlib.metadata as md,json,shutil,subprocess
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect
from scripts.train_then_burn import APPROVED_BURNS,BURN_HASH,GUARD_HASH,digest,approved_launcher
from accelerate.commands.config.config_args import default_yaml_config_file
status=inspect();print('INSPECTION',json.dumps(status),flush=True)
assert not status['guard_disabled'], 'GPU guard already held'
assert digest('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
approved={**APPROVED_BURNS,str(Path('resources/llm_pretrain_burn.py').resolve()):BURN_HASH}
for pid in status['workers']:
    parent=status['processes'][str(pid)]['ppid']
    assert approved_launcher(pid,approved) or (parent>1 and approved_launcher(parent,approved)), 'Unknown GPU workload'
print('ENVIRONMENT',json.dumps({p:md.version(p) for p in ['torch','transformers','accelerate','datasets','safetensors']}),flush=True)
subprocess.run(['/mnt/local/conda-py311/envs/train_env/bin/python','-m','pip','check'],check=True)
recipe=json.loads(Path('proxy_heads.b200.json').read_text())
for key in ['data_dir','eval_data_dir']:
    root=Path(recipe[key]);files=sorted(root.glob('*.parquet'))
    assert files,(key,str(root))
    print('DATA_FILES',key,len(files),sum(p.stat().st_size for p in files),str(files[0]),flush=True)
for name in ['config.json','tokenizer.json','tokenizer_config.json']:
    p=Path(recipe['tokenizer_name'])/name;assert p.is_file()
    print('MODEL_ASSET',name,p.stat().st_size,digest(p),flush=True)
print('ACCELERATE_CACHE',default_yaml_config_file,flush=True)
print('OUTPUT_DISK',shutil.disk_usage('/mnt/local/_outputs'),flush=True)
print('BASELINE_PREFLIGHT_VERIFIED',flush=True)
PY
