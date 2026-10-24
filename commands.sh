#1 +60+a
#th2-6y3o-dot-stop-optimized-launch-20261025-a02
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline NCCL_NVLS_ENABLE=0 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/train/bin/python3.11 scripts/stop_optimized_launch.py launch \
 --root /mnt/local/_outputs/deep-llms_th2/6y3o-dot-stop-optimized-20261025-a02 \
 --parent /mnt/local/_outputs/deep-llms_th2/6y3o-dot-long-pair-20261023-a01 \
 --previous /mnt/local/_outputs/deep-llms_th2/6y3o-dot-stop-optimized-20261025-a01 \
 --smoke-previous /mnt/local/_outputs/deep-llms_th2/6y3o-dot-stop-v2-20261024-a01 \
 --host thiennh-p6-6y3o-worker-0 --session dot_stop_optimized_20261025_a02
