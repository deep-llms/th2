#1 +30+a
#th2-6y3o-fa2-offline-build-20261019-a02
set -euo pipefail
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
TASK_PYTHON=/mnt/local/conda-py311/envs/train_fa2/bin/python3.11
test -x "$TASK_PYTHON"
date -u
hostname
"$TASK_PYTHON" -u scripts/install_fa2_offline.py \
  --archive /mnt/local/_data/deep-llms_th2/build_inputs/flash_attn-2.8.3.post2.tar.gz \
  --work-dir /mnt/local/_outputs/deep-llms_th2/6y3o-fa2-build-20261019-a02
