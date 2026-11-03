#1 +30+a
#th2-wc69-depth-restore-inputs-20261103-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
TASK_PYTHON=/mnt/local/conda-py311/envs/a2_env/bin/python3.11
"$TASK_PYTHON" -u -m scripts.a2_depth_wc69_restore --recipe resources/a2_depth_wc69_restore_20261103.json
"$TASK_PYTHON" -u - <<'CHECK'
from datetime import datetime,timezone
from scripts.gpu_process import inspect
import json
print('RESTORE_FINISHED',datetime.now(timezone.utc).isoformat(),flush=True)
print('READONLY_OWNERSHIP',json.dumps(inspect()),flush=True)
CHECK
