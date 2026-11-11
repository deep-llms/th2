#1 +30+a
#th2-wc69-dot-pool-data-prepare-20261112-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4
TASK_PYTHON=/mnt/local/conda-py311/envs/a2_env/bin/python3.11
TASK_OUT=/mnt/local/_outputs/deep-llms_th2/dot_pool_20261112
mkdir -p "$TASK_OUT"
test ! -e "$TASK_OUT/data"
"$TASK_PYTHON" -u -m scripts.dot_pool_prepare --output "$TASK_OUT/data"
