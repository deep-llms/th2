#1 +60+a
#th2-wc69-dot-rl-v12-launch-20261114-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=2
TASK_PYTHON=/mnt/local/conda-py311/envs/dot_pool/bin/python3.11
test -x "$TASK_PYTHON"
"$TASK_PYTHON" -u -m scripts.launch_dot_rl --commit 5f6c0d7ae2a5c0b8d4c4cfa7819865ee858af5d6
