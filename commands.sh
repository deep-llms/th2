#1 +60+a
#th2-q359-verify-envs-20261009-a01
set -euo pipefail
date -u
hostname
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline WANDB_DISABLED=true OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
status=0
for task_env in train_env eval attention_bench eval_fa4; do
  task_python="/mnt/local/conda-py311/envs/$task_env/bin/python3.11"
  if ! "$task_python" -u - "$task_env" <<'PYENV'
import importlib
import importlib.metadata as md
from pathlib import Path
import sys
from packaging.requirements import Requirement
import torch
name = sys.argv[1]
print('ENV_BEGIN', name, sys.executable, sys.version, flush=True)
assert sys.version_info[:2] == (3, 11)
assert Path(sys.prefix).name == name, sys.prefix
errors = []
for line in Path('envs', name + '.txt').read_text().splitlines():
    if not line.strip() or line.startswith('#'):
        continue
    req = Requirement(line)
    try:
        version = md.version(req.name)
        print('PACKAGE', req.name, version, flush=True)
        assert req.specifier.contains(version), (req.name, version, str(req.specifier))
    except Exception as exc:
        errors.append(str(exc))
modules = ['torch', 'transformers', 'datasets', 'accelerate']
modules += (['lm_eval', 'lm_eval.models.huggingface', 'sentencepiece'] if name in ('eval', 'eval_fa4') else
            ['scipy', 'wandb', 'pyarrow', 'huggingface_hub', 'openai', 'train', 'deep_kv.proxy_training'])
if name in ('attention_bench', 'eval_fa4'):
    modules += ['cutlass', 'flash_attn.cute.interface']
for module in modules:
    try:
        imported = importlib.import_module(module)
        if module == 'flash_attn.cute.interface':
            assert callable(imported.flash_attn_varlen_func)
        print('IMPORT_OK', module, flush=True)
    except Exception as exc:
        errors.append(f'{module}: {type(exc).__name__}: {exc}')
try:
    from transformers import Trainer, Qwen3ForCausalLM
    assert torch.cuda.is_available()
    assert torch.distributed.is_available() and torch.distributed.is_nccl_available()
    print('CUDA', torch.version.cuda, 'DEVICES', torch.cuda.device_count(), 'ARCHES', torch.cuda.get_arch_list(), flush=True)
except Exception as exc:
    errors.append(f'Runtime: {type(exc).__name__}: {exc}')
if errors:
    for error in errors:
        print('ENV_ERROR', name, error, flush=True)
    raise SystemExit(1)
print('ENV_VERIFIED', name, flush=True)
PYENV
  then
    status=1
  fi
  if ! "$task_python" -m pip check; then
    status=1
  fi
done
if [ "$status" -ne 0 ]; then
  echo ENV_VERIFICATION_FAILED
  exit "$status"
fi
echo ALL_FOUR_ENVS_VERIFIED
