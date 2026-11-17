#1 +30+a
#th2-ohmv-a2-runtime-check-20261118-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=disabled
export PYTORCH_NVML_BASED_CUDA_CHECK=1
printf 'A2_RUNTIME_CHECK_BEGIN\n'
date -u
hostname
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name --format=csv,noheader
TASK_RC=0
for TASK_ENV in a2_env eval; do
  TASK_PYTHON="/mnt/local/conda-py311/envs/$TASK_ENV/bin/python3.11"
  if ! test -x "$TASK_PYTHON"; then
    printf 'MISSING_INTERPRETER %s\n' "$TASK_PYTHON"
    TASK_RC=1
    continue
  fi
  if ! "$TASK_PYTHON" -m pip check; then TASK_RC=1; fi
  if ! "$TASK_PYTHON" - "$TASK_ENV" <<'PYVERIFY'
import hashlib, importlib, importlib.metadata as md, json, socket, sys
from pathlib import Path
from packaging.requirements import Requirement
name=sys.argv[1]
p=Path('envs')/(name+'.txt')
result={'env':name,'hostname':socket.gethostname(),'python':sys.version,'executable':sys.executable,'recipe_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'packages':{},'errors':[],'gpu_workload_executed':False}
if sys.version_info[:2]!=(3,11): result['errors'].append('Python is not 3.11')
for line in p.read_text().splitlines():
    if not line.strip() or line.startswith('#'): continue
    req=Requirement(line)
    try:
        version=md.version(req.name); result['packages'][req.name]=version
        if req.specifier and not req.specifier.contains(version): result['errors'].append(f'{req.name}: {version} violates {req.specifier}')
    except Exception as e: result['errors'].append(f'{req.name}: {e}')
mods=['torch','transformers','datasets','tokenizers','numpy','safetensors','pyarrow','zstandard']
mods+=['accelerate','matplotlib','huggingface_hub'] if name=='a2_env' else ['lm_eval']
for module in mods:
    try: importlib.import_module(module)
    except Exception as e: result['errors'].append(f'import {module}: {type(e).__name__}: {e}')
try:
    import torch
    result['torch_cuda_build']=torch.version.cuda
    result['cuda_available']=torch.cuda.is_available()
    result['gpu_count']=torch.cuda.device_count()
    if not result['cuda_available']: result['errors'].append('CUDA unavailable')
except Exception as e: result['errors'].append(f'CUDA detection: {e}')
result['status']='PASS' if not result['errors'] else 'FAIL'
print('A2_RUNTIME_RESULT '+json.dumps(result,sort_keys=True),flush=True)
print('A2_PIP_FREEZE '+json.dumps(sorted((d.metadata['Name']+'=='+d.version) for d in md.distributions() if d.metadata['Name'])),flush=True)
sys.exit(bool(result['errors']))
PYVERIFY
  then TASK_RC=1; fi
done
printf 'A2_RUNTIME_CHECK_EXIT=%s\n' "$TASK_RC"
exit "$TASK_RC"
