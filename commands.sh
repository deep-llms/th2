#1 +60+a
#th2-joint-inspect-b200-20260923-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true
TASK_PYTHON=/mnt/local/conda-py311/envs/train_env/bin/python3.11
date -u
hostname
git rev-parse HEAD
test -x "$TASK_PYTHON"
"$TASK_PYTHON" -u scripts/inspect_b200.py
