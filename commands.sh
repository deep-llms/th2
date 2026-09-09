#1 +30+a
#th2-swt-readonly-next-capacity-gpu-status-20260909-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate swt
test "$CONDA_DEFAULT_ENV" = swt
TASK_PYTHON=/mnt/local/conda-py311/envs/swt/bin/python
test "$(command -v python)" = "$TASK_PYTHON"
date -u
"$TASK_PYTHON" -m scripts.gpu_status --gpus 0 1 2 3 4 5 6 7
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
echo READONLY_TRAINING_GPU_STATUS_COMPLETE
