#!/usr/bin/env bash
# Source training must already be stopped through its verified supervisor.
set -euo pipefail
TASK_ROOT=${1:?Pass fresh evaluation root}
TASK_SOURCE=${2:?Pass stopped priority queue root}
TASK_REFERENCE=${3:?Pass established benchmark reference root}
TASK_HOST=${4:?Pass verified hostname}
TASK_EARLIER=${5:?Pass original tuning queue root}
cd /mnt/local/deep-llms_th2
test "$(hostname)" = "$TASK_HOST"
test ! -e "$TASK_ROOT"
test -s "$TASK_SOURCE/user-stop-for-evaluation.json"
cat "$TASK_SOURCE/user-stop-for-evaluation.json"
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate eval_fa4
export NCCL_NVLS_ENABLE=0 WANDB_MODE=offline WANDB_PROJECT=deep2shallow
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -u -m scripts.p7_completed_evaluation --root "$TASK_ROOT" --source "$TASK_SOURCE" \
  --reference "$TASK_REFERENCE" --host "$TASK_HOST" --stop-receipt user-stop-for-evaluation.json \
  --tuning-sources "$TASK_EARLIER/supervised/run" "$TASK_SOURCE/supervised/run"
exec python -u -m scripts.train_then_burn --config "$TASK_ROOT/jobs.json" --output "$TASK_ROOT/supervised" \
  --inspection "$TASK_ROOT/inspection.json" --host "$TASK_HOST" --burn-session "$(basename "$TASK_ROOT")-final-burn"
