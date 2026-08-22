#1 +60+a
#th2-check-gpu-burn-runtime-20260822
set -euo pipefail

echo '=== th2 GPU burn runtime check ==='
date -u
hostname

TASK_BURN_SCRIPT="/tmp/llm_pretrain_burn.py"
test -s "$TASK_BURN_SCRIPT"
ls -l "$TASK_BURN_SCRIPT"
sha256sum "$TASK_BURN_SCRIPT"

echo '=== system Python ==='
/usr/bin/python3 --version
/usr/bin/python3 - <<'PY'
import importlib.util
import sys

print('executable', sys.executable)
print('torch_module_available', importlib.util.find_spec('torch') is not None)
print('numpy_module_available', importlib.util.find_spec('numpy') is not None)
PY

echo '=== CUDA tools ==='
command -v nvidia-smi
command -v nvcc || true
command -v gpu-burn || true
command -v gpu_burn || true

echo '=== GPU state ==='
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader

echo 'TH2 GPU BURN RUNTIME CHECK COMPLETE'
