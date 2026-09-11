#1 +120+a
#th2-verify-train-env-and-eval-20260911-a01
set -euo pipefail
date -u
hostname
source /mnt/local/conda-py311/etc/profile.d/conda.sh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_NVML_BASED_CUDA_CHECK=1
for TASK_ENV in train_env eval; do
    conda activate "$TASK_ENV"
    test "$CONDA_DEFAULT_ENV" = "$TASK_ENV"
    test "$(command -v python)" = "/mnt/local/conda-py311/envs/$TASK_ENV/bin/python"
    python -B - "$TASK_ENV" <<'PY'
import importlib, importlib.metadata as md, json, sys
from pathlib import Path
name=sys.argv[1]
assert sys.version_info[:2]==(3,11)
assert Path(sys.prefix)==Path('/mnt/local/conda-py311/envs')/name
expected={'transformers':'5.9.0','datasets':'4.8.5','accelerate':'1.13.0'}
if name=='eval': expected['lm_eval']='0.4.10'
for package,version in expected.items():
    assert md.version(package)==version, (package,md.version(package),version)
modules=['torch','transformers','datasets','accelerate']
modules += ['scipy','wandb','pyarrow','huggingface_hub','openai','entmax'] if name=='train_env' else ['lm_eval','sentencepiece']
for module in modules: importlib.import_module(module)
from transformers import Trainer, TrainingArguments, Qwen3ForCausalLM
if name=='eval':
    from lm_eval.models.huggingface import HFLM
import torch
assert torch.version.cuda is not None
assert torch.cuda.is_available() and torch.cuda.device_count()==8
assert torch.distributed.is_nccl_available()
print('ENV_IMPORT_CHECK_PASSED',json.dumps(dict(env=name,python=sys.executable,
    versions={p:md.version(p) for p in expected},torch=torch.__version__,
    cuda_build=torch.version.cuda,visible_gpus=torch.cuda.device_count(),nccl=True)),flush=True)
PY
    python -m pip check
done
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
date -u
echo TRAIN_ENV_AND_EVAL_VERIFIED
