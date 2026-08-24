#1 +60+a
#th2-verify-eval-env-with-burns-running-20260824
#!/usr/bin/env bash
set -euo pipefail

echo '=== verify fresh eval environment without touching GPU workloads ==='
date -u
hostname

TASK_PYTHON=/mnt/local/conda-py311/envs/eval/bin/python3.11
test -x "$TASK_PYTHON"

mapfile -t TASK_GPU_PIDS_BEFORE < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | sed '/^[[:space:]]*$/d' | sort -n
)
echo "gpu_compute_processes_before=${#TASK_GPU_PIDS_BEFORE[@]}"
test "${#TASK_GPU_PIDS_BEFORE[@]}" -eq 8
printf 'pid=%s\n' "${TASK_GPU_PIDS_BEFORE[@]}"

"$TASK_PYTHON" - <<'PY'
import importlib.metadata
import importlib.util

import accelerate
import datasets
import entmax
import lm_eval
import pyarrow
import scipy
import sentencepiece
import torch
import transformers
from lm_eval.models.huggingface import HFLM

from eval.benchmarks import TASK_CONFIGS

versions = {
    "python_package_lm_eval": importlib.metadata.version("lm_eval"),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "datasets": datasets.__version__,
    "accelerate": accelerate.__version__,
    "entmax": importlib.metadata.version("entmax"),
    "scipy": scipy.__version__,
    "pyarrow": pyarrow.__version__,
    "sentencepiece": importlib.metadata.version("sentencepiece"),
}
assert versions["python_package_lm_eval"] == "0.4.10", versions
assert versions["transformers"] == "5.9.0", versions
assert versions["datasets"] == "4.8.5", versions
assert versions["accelerate"] == "1.13.0", versions
assert importlib.util.find_spec("fasttext") is None, "fasttext must not be installed in eval"
tasks = [task for group in TASK_CONFIGS.values() for task in group]
assert len(tasks) == 26, len(tasks)
assert len(tasks) == len(set(tasks)), "duplicate evaluation tasks"
assert HFLM is not None and lm_eval is not None and entmax is not None
for name, version in versions.items():
    print(f"{name}={version}")
print(f"configured_task_groups={len(TASK_CONFIGS)} configured_tasks={len(tasks)}")
print("fasttext=ABSENT")
print("EVAL_IMPORTS_OK")
PY

mapfile -t TASK_GPU_PIDS_AFTER < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | sed '/^[[:space:]]*$/d' | sort -n
)
test "${TASK_GPU_PIDS_BEFORE[*]}" = "${TASK_GPU_PIDS_AFTER[*]}"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
echo 'TH2 EVAL ENV VERIFIED; GPU BURNS UNCHANGED'
