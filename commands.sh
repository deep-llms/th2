#1 +90+a
#th2-verify-eval-and-sparse-emb-fresh-pod-20260902-a16
set -euo pipefail

TASK_TRAIN_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11
TASK_EVAL_PYTHON=/mnt/local/conda-py311/envs/eval/bin/python3.11
TASK_BURN_SCRIPT=/tmp/llm_pretrain_burn.py

gpu_pids() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
}

echo '=== immutable runtime verification preflight ==='
date -u
hostname
test -x "$TASK_TRAIN_PYTHON"
test -x "$TASK_EVAL_PYTHON"
test -s "$TASK_BURN_SCRIPT"

mapfile -t TASK_BEFORE_PIDS < <(gpu_pids)
[[ "${#TASK_BEFORE_PIDS[@]}" -eq 8 ]]
mapfile -t TASK_BEFORE_UUIDS < <(
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader \
        | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | awk 'NF' | sort -u
)
[[ "${#TASK_BEFORE_UUIDS[@]}" -eq 8 ]]

TASK_BURN_LAUNCHER=''
for TASK_PID in "${TASK_BEFORE_PIDS[@]}"; do
    [[ "$TASK_PID" =~ ^[0-9]+$ && "$TASK_PID" != 1 ]]
    test -r "/proc/$TASK_PID/status"
    TASK_PARENT="$(awk '/^PPid:/ {print $2}' "/proc/$TASK_PID/status")"
    if [[ -z "$TASK_BURN_LAUNCHER" ]]; then
        TASK_BURN_LAUNCHER="$TASK_PARENT"
    else
        [[ "$TASK_PARENT" == "$TASK_BURN_LAUNCHER" ]]
    fi
done
[[ "$TASK_BURN_LAUNCHER" =~ ^[0-9]+$ && "$TASK_BURN_LAUNCHER" != 1 ]]
TASK_BURN_COMMAND="$(tr '\0' ' ' < "/proc/$TASK_BURN_LAUNCHER/cmdline")"
[[ "$TASK_BURN_COMMAND" == *"$TASK_BURN_SCRIPT"* ]]
echo "VERIFIED_EXISTING_BURN launcher=$TASK_BURN_LAUNCHER workers=${TASK_BEFORE_PIDS[*]}"

echo '=== sparse_emb runtime ==='
"$TASK_TRAIN_PYTHON" - <<'PY'
import importlib.metadata
import sys

import accelerate
import datasets
import entmax
import huggingface_hub
import numpy
import openai
import pyarrow
import scipy
import torch
import transformers
import wandb

assert sys.executable == "/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11", sys.executable
assert sys.version_info[:2] == (3, 11), sys.version
assert transformers.__version__ == "5.9.0"
assert datasets.__version__ == "4.8.5"
assert accelerate.__version__ == "1.13.0"
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 8
assert torch.cuda.is_bf16_supported()
names = [torch.cuda.get_device_name(index) for index in range(8)]
assert all("B200" in name for name in names), names
for index in range(8):
    value = torch.ones(1, device=f"cuda:{index}", dtype=torch.bfloat16) + 1
    assert value.item() == 2.0
torch.cuda.synchronize()
print(f"python={sys.version.split()[0]}")
print(f"training_env=OK torch={torch.__version__} cuda={torch.version.cuda}")
print(f"transformers={transformers.__version__} datasets={datasets.__version__} accelerate={accelerate.__version__}")
print(f"entmax={importlib.metadata.version('entmax')} pyarrow={pyarrow.__version__}")
print(f"gpu_names={names}")
PY

echo '=== eval runtime ==='
"$TASK_EVAL_PYTHON" - <<'PY'
import importlib.metadata
import importlib.util
import sys

import accelerate
import datasets
import entmax
import lm_eval
import pyarrow
import scipy
import sentencepiece
import torch
import transformers

assert sys.executable == "/mnt/local/conda-py311/envs/eval/bin/python3.11", sys.executable
assert sys.version_info[:2] == (3, 11), sys.version
assert transformers.__version__ == "5.9.0"
assert datasets.__version__ == "4.8.5"
assert accelerate.__version__ == "1.13.0"
assert importlib.metadata.version("lm_eval") == "0.4.10"
assert importlib.metadata.version("entmax") == "1.3"
assert importlib.util.find_spec("fasttext") is None
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 8
assert torch.cuda.is_bf16_supported()
names = [torch.cuda.get_device_name(index) for index in range(8)]
assert all("B200" in name for name in names), names
for index in range(8):
    value = torch.ones(1, device=f"cuda:{index}", dtype=torch.bfloat16) + 1
    assert value.item() == 2.0
torch.cuda.synchronize()
print(f"python={sys.version.split()[0]}")
print(f"eval_env=OK torch={torch.__version__} cuda={torch.version.cuda}")
print(f"transformers={transformers.__version__} datasets={datasets.__version__} accelerate={accelerate.__version__}")
print(f"lm_eval={importlib.metadata.version('lm_eval')} entmax={importlib.metadata.version('entmax')} fasttext=ABSENT")
print(f"gpu_names={names}")
PY

echo '=== prove burn ownership and coverage are unchanged ==='
mapfile -t TASK_AFTER_PIDS < <(gpu_pids)
[[ "${TASK_AFTER_PIDS[*]}" == "${TASK_BEFORE_PIDS[*]}" ]]
mapfile -t TASK_AFTER_UUIDS < <(
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader \
        | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | awk 'NF' | sort -u
)
[[ "${TASK_AFTER_UUIDS[*]}" == "${TASK_BEFORE_UUIDS[*]}" ]]
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits
echo 'TH2 EVAL AND SPARSE_EMB ENVIRONMENTS VERIFY OK; BURNS UNCHANGED'
