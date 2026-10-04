#1 +60+a
#th2-tjx3-verify-environments-20261004-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline NCCL_NVLS_ENABLE=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
hostname
git rev-parse HEAD
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
python3 -u - <<'PY'
import json, pathlib, subprocess
from datetime import datetime, timezone
print('ENV_AUDIT_AT', datetime.now(timezone.utc).isoformat(), flush=True)
checks = []
probe = r'''
import importlib, importlib.metadata, json, pathlib, sys
name = sys.argv[1]
print('INTERPRETER', name, sys.executable, sys.version, flush=True)
assert sys.version_info[:2] == (3,11)
packages = ['torch', 'transformers', 'datasets', 'accelerate', 'huggingface_hub', 'numpy', 'pyarrow']
packages += ['scipy', 'wandb', 'openai', 'entmax'] if name == 'train_env' else ['lm_eval', 'sentencepiece']
print('PACKAGE_VERSIONS', name, json.dumps({p: importlib.metadata.version(p) for p in packages}), flush=True)
for p in packages:
    importlib.import_module(p)
for p, expected in [('transformers','5.9.0'), ('datasets','4.8.5'), ('accelerate','1.13.0')]:
    assert importlib.metadata.version(p) == expected, p
import torch
assert torch.version.cuda is not None and torch.distributed.is_nccl_available()
print('TORCH_BUILD', name, json.dumps(dict(version=torch.__version__, cuda=torch.version.cuda,
    arch_flags=torch._C._cuda_getArchFlags(), nccl_available=torch.distributed.is_nccl_available())), flush=True)
from transformers import Trainer, Qwen3ForCausalLM
if name == 'train_env':
    import train
    from deep_kv.model import DeepKV
    from deep_kv.training import DeepKVTrainer
else:
    from lm_eval.models.huggingface import HFLM
    assert importlib.metadata.version('lm_eval') == '0.4.10'
from accelerate.commands.config.config_args import default_yaml_config_file
source = pathlib.Path('resources/accelerate_config.yaml')
cached = pathlib.Path(default_yaml_config_file)
print('ACCELERATE_CACHE', name, json.dumps(dict(path=str(cached), exists=cached.is_file(),
    matches_resource=cached.read_bytes()==source.read_bytes() if cached.is_file() else False)), flush=True)
print('IMPORT_CHECK_PASSED', name, flush=True)
'''
for name in ('train_env','eval'):
    python = pathlib.Path('/mnt/local/conda-py311/envs') / name / 'bin/python'
    if not python.is_file():
        print('MISSING_ENVIRONMENT', str(python), flush=True)
        checks.append(1)
        continue
    for label, argv in [('imports', [str(python), '-u', '-c', probe, name]),
                        ('pip_check', [str(python), '-m', 'pip', 'check']),
                        ('accelerate_env', [str(python), '-m', 'accelerate.commands.accelerate_cli', 'env'])]:
        result = subprocess.run(argv, timeout=180)
        checks.append(result.returncode)
        print('CHECK_RETURN_CODE', name, label, result.returncode, flush=True)
    if name == 'train_env':
        result = subprocess.run([str(python), '-m', 'unittest',
            'tests.test_deep_kv.DeepKVAcceptance.test_base_equivalence_and_identical_initialization', '-v'], timeout=180)
        checks.append(result.returncode)
        print('CPU_MODEL_CHECK_RETURN_CODE', result.returncode, flush=True)
print('ENVIRONMENT_AUDIT_RESULT', json.dumps(dict(passed=not any(checks), returncodes=checks)), flush=True)
raise SystemExit(1 if any(checks) else 0)
PY
