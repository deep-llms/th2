#!/usr/bin/env bash
# Only launch after explicit authorization to stop the identified source queue.
set -euo pipefail
TASK_ROOT=${1:?Pass fresh evaluation root}
TASK_SOURCE=${2:?Pass authorized source queue root}
TASK_REFERENCE=${3:?Pass existing remaining-27 evaluation root}
TASK_HOST=${4:?Pass verified hostname}
cd /mnt/local/deep-llms_th2
test "$(hostname)" = "$TASK_HOST"
test ! -e "$TASK_ROOT"
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate eval_fa4
export NCCL_NVLS_ENABLE=0 WANDB_MODE=offline WANDB_PROJECT=deep2shallow
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Validate completion before signaling; this computes no GPU work.
python -u - "$TASK_SOURCE" "$TASK_REFERENCE" <<'PY'
import sys
from pathlib import Path
from scripts.p7_completed_evaluation import build_scope
scope, _ = build_scope(Path(sys.argv[1]) / 'supervised/run', sys.argv[2])
print('VERIFIED_COMPLETED_CHECKPOINTS', scope['completed_labels'], flush=True)
PY
python -u -m scripts.stop_proxy_followup --root "$TASK_SOURCE" --session "$(basename "$TASK_SOURCE")" \
  --host "$TASK_HOST" --burn-session "$(basename "$TASK_SOURCE")-final-burn" \
  --output "$TASK_SOURCE/user-stop-after-keyonly.json"
python -u -m scripts.p7_completed_evaluation --root "$TASK_ROOT" --source "$TASK_SOURCE" \
  --reference "$TASK_REFERENCE" --host "$TASK_HOST"
exec python -u -m scripts.train_then_burn --config "$TASK_ROOT/jobs.json" --output "$TASK_ROOT/supervised" \
  --inspection "$TASK_ROOT/inspection.json" --host "$TASK_HOST" --burn-session "$(basename "$TASK_ROOT")-final-burn"
