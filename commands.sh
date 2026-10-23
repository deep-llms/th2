#1 +60+a
#th2-6y3o-dot-downstream-launch-20261023-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline NCCL_NVLS_ENABLE=0 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/eval/bin/python3.11 scripts/bdot_downstream_launch.py \
  --parent /mnt/local/_outputs/deep-llms_th2/6y3o-dot-long-pair-20261023-a01 \
  --root /mnt/local/_outputs/deep-llms_th2/6y3o-dot-downstream-20261023-a01 \
  --data-root /mnt/local/_data/deep-llms_th2/benchmarks_zero_20261023 \
  --host thiennh-p6-6y3o-worker-0 --session dot_downstream_20261023_a01
