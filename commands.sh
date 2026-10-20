#1 +30+a
#th2-6y3o-dot-grouped-verify-20261020-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/train/bin/python3.11 scripts/bdot_b200_launch.py \
 --root /mnt/local/_outputs/deep-llms_th2/6y3o-dot-grouped-20261020-a01 \
 --session dot_grouped_20261020_a01 \
 --host thiennh-p6-6y3o-worker-0 \
 --jobs-config resources/bdot_b200_grouped.json
