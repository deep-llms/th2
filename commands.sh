#1 +60+a
#th2-readonly-discover-diagnostic-python-20260902-a07
set -euo pipefail

echo '=== read-only Python and Conda inventory; GPU burns are not modified ==='
date -u
hostname
pwd
command -v python || true
command -v python3 || true
command -v conda || true

mapfile -t TASK_PYTHONS < <(
    {
        command -v python 2>/dev/null || true
        command -v python3 2>/dev/null || true
        find /mnt/local -maxdepth 6 \( -type f -o -type l \) \
            \( -name python -o -name python3 -o -name python3.11 \) \
            -perm /111 -print 2>/dev/null || true
    } | sort -u
)

printf 'python_candidates=%s\n' "${#TASK_PYTHONS[@]}"
for TASK_PYTHON in "${TASK_PYTHONS[@]}"; do
    echo "--- $TASK_PYTHON ---"
    "$TASK_PYTHON" --version 2>&1 || true
    "$TASK_PYTHON" - <<'PY' 2>&1 || true
import importlib.metadata
import importlib.util

for name in ("torch", "transformers", "datasets", "numpy", "tqdm"):
    spec = importlib.util.find_spec(name)
    try:
        version = importlib.metadata.version(name) if spec else None
    except Exception as error:
        version = f"ERROR:{error!r}"
    print(f"{name}: present={spec is not None} version={version}")
if importlib.util.find_spec("torch"):
    import torch
    print(f"cuda_available={torch.cuda.is_available()} gpu_count={torch.cuda.device_count()}")
PY
done

echo '=== GPU state unchanged ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits
echo 'TH2 READONLY DIAGNOSTIC PYTHON INVENTORY COMPLETE'
