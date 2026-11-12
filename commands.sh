#1 +30+a
#th2-wc69-dot-pool-e1b-v15-20261113-a02
set -euo pipefail
TASK_PYTHON=/mnt/local/conda-py311/envs/dot_pool/bin/python3.11
test -x "$TASK_PYTHON"
export DOT_POOL_SPEC_SHA256=41aea5e294682e4ee703cff1df4e95a2de84662b7eba880eb862b08067ff2065
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=2
"$TASK_PYTHON" -u -m scripts.launch_dot_pool_e1b --commit 914fb85dd73eec3da20457a23745bb6cb4a7f531
