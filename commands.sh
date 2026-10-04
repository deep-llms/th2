#1 +60+a
#th2-tjx3-document-isolation-20261004-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline NCCL_NVLS_ENABLE=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
TASK_PYTHON=/mnt/local/conda-py311/envs/train_env/bin/python
hostname
date -u
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv
TASK_ROOT=/mnt/local/_outputs/@PROJECT@/document-isolation-20261004-a01
mkdir "$TASK_ROOT"
python3 -m scripts.verified_gpu_reclaim inspect --output "$TASK_ROOT/gpu-inspection.json"
"$TASK_PYTHON" -u - <<'CHECK'
import json, pathlib, shutil, importlib.metadata
from accelerate.commands.config.config_args import default_yaml_config_file
print('VERSIONS',json.dumps({p:importlib.metadata.version(p) for p in ['torch','transformers','accelerate','datasets']}),flush=True)
source=pathlib.Path('resources/accelerate_config.yaml')
dest=pathlib.Path(default_yaml_config_file)
dest.parent.mkdir(parents=True,exist_ok=True)
if dest.exists() and dest.read_bytes()!=source.read_bytes():
    backup=dest.with_name(dest.name+'.before-document-isolation-20261004-a01')
    with backup.open('xb') as f:f.write(dest.read_bytes())
shutil.copy2(source,dest)
assert source.read_bytes()==dest.read_bytes()
print('ACCELERATE_CONFIG_VERIFIED',str(dest),flush=True)
from train import load_text
recipe=json.loads(pathlib.Path('deep_kv.b200.json').read_text())
for key,count in [('data_dir',36595514),('eval_data_dir',11822)]:
    data=load_text(recipe[key]);assert len(data)==count and data.column_names==['text']
    print('B200_TRAIN_LOADER_OK',key,len(data),flush=True)
CHECK
"$TASK_PYTHON" -m accelerate.commands.accelerate_cli env
python3 scripts/gpu_status.py --gpus 0 --require-free
export CUDA_VISIBLE_DEVICES=0 DOCUMENT_TEST_DEVICE=cuda:0
DOCUMENT_TEST_BF16=0 "$TASK_PYTHON" -m unittest tests.test_document_isolation -v
DOCUMENT_TEST_BF16=1 "$TASK_PYTHON" -m unittest tests.test_document_isolation -v
"$TASK_PYTHON" -m scripts.benchmark_document_attention
python3 scripts/gpu_status.py --gpus 0 --require-free
echo B200_DOCUMENT_ISOLATION_TESTS_PASSED
