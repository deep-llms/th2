#1 +30+a
#th2-wc69-dot-pool-sr-diagnostics-correction-20261112-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
TASK_PYTHON=/mnt/local/conda-py311/envs/dot_pool/bin/python3.11
TASK_COMMIT=$(git rev-parse HEAD)
"$TASK_PYTHON" -u -m scripts.launch_dot_pool_sr_recheck --commit "$TASK_COMMIT"
