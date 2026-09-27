#1 +60+a
#th2-78gg-deep-kv-smoke-preflight-20260927-a01
set -euo pipefail
date -u
hostname
test "$(hostname)" = thiennh-p6-78gg-worker-0
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate train_env
test "$CONDA_DEFAULT_ENV" = train_env
TASK_OUTPUT=/mnt/local/_outputs/@PROJECT@/deep-kv-smoke-preflight-20260927-a01
test ! -e "$TASK_OUTPUT"
mkdir -p "$TASK_OUTPUT"
exec > >(tee "$TASK_OUTPUT/preflight.log") 2>&1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline HF_HUB_DISABLE_TELEMETRY=1
python -u - "$TASK_OUTPUT" <<'PY'
import hashlib,json,shutil,sys
from pathlib import Path
import accelerate,datasets,torch,transformers
from accelerate.commands.config.config_args import default_yaml_config_file,load_config_from_file
from deep_kv.config import load_config
from deep_kv.data import load_text
from scripts.gpu_status import snapshot
output=Path(sys.argv[1])
print('RUNTIME',json.dumps({'python':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'transformers':transformers.__version__,'accelerate':accelerate.__version__,'datasets':datasets.__version__}),flush=True)
assert transformers.__version__=='5.9.0' and accelerate.__version__=='1.13.0'
gpus=snapshot(list(range(8)))
assert len(gpus)==8 and torch.cuda.device_count()==8
print('GPU_STATUS',json.dumps(gpus),flush=True)
config=load_config('deep_kv.b200.json')
for key in ('model_config','tokenizer','train_data','eval_data'):
    assert Path(config[key]).exists(),key
for split in ('train','eval'):
    ds=load_text(config[f'{split}_data'])
    assert ds.column_names==['text'] and len(ds)>0
    print('SAMPLED_DATA',split,len(ds),ds._fingerprint,flush=True)
source=Path('resources/accelerate_config.yaml')
dest=Path(default_yaml_config_file)
dest.parent.mkdir(parents=True,exist_ok=True)
if dest.exists():
    shutil.copy2(dest,output/'accelerate_config.before.yaml')
shutil.copy2(source,dest)
assert dest.read_bytes()==source.read_bytes()
settings=load_config_from_file(str(dest)).to_dict()
assert settings['num_processes']==8 and settings['mixed_precision']=='bf16' and settings['distributed_type']=='MULTI_GPU'
print('ACCELERATE_CONFIG_INSTALLED',str(dest),hashlib.sha256(dest.read_bytes()).hexdigest(),flush=True)
for path in (Path('/tmp/llm_pretrain_burn.py'),Path('/mnt/local/_gpu_guard/polite_burn.py')):
    if path.is_file():
        body=path.read_bytes()
        print('BURN_SOURCE',str(path),hashlib.sha256(body).hexdigest(),flush=True)
        if len(body)<65536: print(body.decode(),flush=True)
print('GUARD_DISABLED',Path('/mnt/local/_gpu_guard/DISABLED').exists(),flush=True)
PY
accelerate env
python -m scripts.verified_gpu_reclaim inspect --output "$TASK_OUTPUT/gpu_inspection.json"
df -h /mnt/local
printf '%s\n' 'DEEP_KV_SMOKE_PREFLIGHT_OK'
