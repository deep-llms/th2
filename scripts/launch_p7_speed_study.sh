#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=${1:?Pass fresh output root}
TASK_HOST=${2:?Pass verified hostname}
TASK_PREDECESSOR=${3:?Pass completed predecessor root}
cd /mnt/local/deep-llms_th2
test "$(hostname)" = "$TASK_HOST"
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate attention_bench
export NCCL_NVLS_ENABLE=0 WANDB_MODE=offline WANDB_PROJECT=deep2shallow
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TASK_EXTRA=()
if [[ ${4:-profile} == trial ]]; then
  TASK_EXTRA=(--trial)
  export TORCHINDUCTOR_CACHE_DIR="$TASK_ROOT/inductor" TRITON_CACHE_DIR="$TASK_ROOT/triton"
  export TORCHINDUCTOR_COMPILE_THREADS=1
fi
python -u -m scripts.p7_speed_study prepare --root "$TASK_ROOT" --host "$TASK_HOST" --predecessor "$TASK_PREDECESSOR" "${TASK_EXTRA[@]}"
exec python -u -m scripts.train_then_burn --config "$TASK_ROOT/jobs.json" --output "$TASK_ROOT/supervised" \
  --inspection "$TASK_ROOT/inspection.json" --host "$TASK_HOST" --burn-session "$(basename "$TASK_ROOT")-final-burn"
